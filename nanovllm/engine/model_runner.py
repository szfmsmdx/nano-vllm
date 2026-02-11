import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# 挟持函数，给注入 tp_group 参数
def get_patched_dist_fn(origin_fn, tp_group):
    def patched(*args, **kwargs):
        if 'group' not in kwargs and tp_group is not None:
            kwargs['group'] = tp_group
        return  origin_fn(*args, **kwargs)
    return patched

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank    # 全局rank
        self.event = event

        # 初始化逻辑是同步的，当 world size个进程连接到这个 group 队列中才会返回
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)  # 分布式训练组
        torch.cuda.set_device(rank)

        # PD 分离初始化
        self.tp_group = None
        self.pd_role = None # 'prefill' / 'decode'
        self.peer_rank = None # 对应的传输目标/来源

        if config.pd_separation:
            half_size = self.world_size // 2
            if rank < half_size:
                self.pd_role = 'prefill'
                self.peer_rank = rank + half_size
            else:
                self.pd_role = 'decode'
                self.peer_rank = rank - half_size

            group_prefill = dist.new_group(list(range(0, half_size)))
            group_decode = dist.new_group(list(range(half_size, self.world_size)))

            self.tp_group = group_prefill if self.pd_role == 'prefill' else group_decode
            self._patch_distributed_ops()
        else:
            self.pd_role = 'None'
            self.tp_group = None

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)    # 自己重新写的支持并行的 Qwen3ForCausalLM
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager and self.pd_role != 'prefill':      # True 则走普通 torch 代码
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 进程间通信
        if self.world_size > 1:
            if rank == 0:   # 主进程逻辑
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)   # 1MB
                dist.barrier()  # 等待所有的分布式进程都到达 barrier，一种同步操作
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")    # 连接主进程的 sharemem
                self.loop()

    def _patch_distributed_ops(self):
        """
        动态替换 dist 操作，使其默认使用子组
        """
        if self.tp_group is None:
            return 
        
        self._orig_all_reduce = dist.all_reduce
        self._orig_get_world_size = dist.get_world_size
        self._orig_get_rank = dist.get_rank

        # 替换为带 group 版本
        dist.all_reduce = get_patched_dist_fn(self._orig_all_reduce, self.tp_group)
        # 返回子组信息
        dist.get_world_size = lambda group=None: self._orig_get_world_size(self.tp_group) if group is None else self._orig_get_world_size(group)
        dist.get_rank = lambda group=None: self._orig_get_rank(self.tp_group) if group is None else self._orig_get_rank(group)


    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:  # rank 0控制子进程
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配当前Runner所处device的kvcache
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        tp_size = dist.get_world_size()
        num_kv_heads = hf_config.num_key_value_heads // tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # block_size：一个block放多少token
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - peak - used + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        layer_id = 0
        for module in self.model.modules:
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # P2P 传输 KV Cache
    def transfer_kv_cache(self, seq_ids: list[int], block_tables_map: dict):
        """
        Prefill 节点调用：发送 KV Cache 到对应的 Peer Decode 节点
        seq_ids: 本轮完成 Prefill 的序列 ID 列表
        block_tables_map: {seq_id: [block_id1, block_id2...]}
        """
        if self.pd_role != 'prefill': return
        for seq_id in seq_ids:
            blocks = block_tables_map[seq_id]
            # 发送元数据
            meta = torch.tensor([seq_id, len(blocks)], dtype=torch.int32).cuda()
            dist.send(meta, dst=self.peer_rank)

            # 发送实际数据
            for layer_idx in range(self.config.hf_config.num_hidden_layers):
                for block_id in blocks:
                    k_data = self.kv_cache[0, layer_idx, block_id]
                    v_data = self.kv_cache[1, layer_idx, block_id]
                    dist.send(k_data, dst=self.peer_rank)
                    dist.send(v_data, dst=self.peer_rank)

    def recive_kv_cache(self, recv_block_tables_map: dict[int, list[int]]) -> dict[int, list[int]]:
        """
        Decode 节点调用：从 Peer Prefill 节点接收 KV Cache
        recv_block_tables_map: 外部提前分配好的 block table，避免 ModelRunner 直接依赖 BlockManager
        返回: {seq_id: [block_id...]} 映射
        """
        if self.pd_role != "decode":
            return {}

        receive_map = {}
        num_seqs_to_recv = len(recv_block_tables_map)
        for _ in range(num_seqs_to_recv):
            meta = torch.empty(2, dtype=torch.int32).cuda()
            dist.recv(meta, src=self.peer_rank)
            seq_id = meta[0].item()
            num_blocks = meta[1].item()

            blocks = recv_block_tables_map.get(seq_id)
            if blocks is None:
                raise KeyError(f"Missing block table for received seq_id={seq_id}")
            if len(blocks) != num_blocks:
                raise ValueError(
                    f"KV block size mismatch for seq_id={seq_id}: expected {num_blocks}, got {len(blocks)}"
                )

            for layer_idx in range(self.config.hf_config.num_hidden_layers):
                for block_id in blocks:
                    k_dst = self.kv_cache[0, layer_idx, block_id]
                    v_dst = self.kv_cache[1, layer_idx, block_id]
                    dist.recv(k_dst, src=self.peer_rank)
                    dist.recv(v_dst, src=self.peer_rank)

            receive_map[seq_id] = blocks
        return receive_map

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 整理成 2D tensor，需要 padding
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill_legacy(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # 累计长度
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])   # 没有被 cache 过的部分 extend 拼接
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))    # 拼接位置
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))    # 针对没有被 cache 部分，slot mapping指明了该存在kv cache的哪里
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache 
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions
    
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q, max_seqlen_k = 0, 0
        slot_mapping = []

        for seq in seqs:
            # 兼容 warm up 使用的非 chunk 全量 prefill
            chunk_size = getattr(seq, "current_chunk_size", len(seq) - seq.num_cached_tokens)
            start_pos = seq.num_cached_tokens
            end_pos = start_pos + chunk_size

            input_ids.extend(seq[start_pos : end_pos])
            positions.extend(list(range(start_pos, end_pos)))

            seqlen_q = chunk_size
            seqlen_k = end_pos
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q, max_seqlen_k = max(seqlen_q, max_seqlen_q), max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue

            # 给token分配 block槽位
            for i in range(start_pos, end_pos):
                b_idx, b_offset = i // self.block_size, i % self.block_size
                slot = seq.block_table[b_idx] * self.block_size + b_offset
                slot_mapping.append(slot)

        # 如果存在已经缓存的 token(cu_seqlens_k > cu_seqlens_q)，那么读取历史 KV
        block_tables = self.prepare_block_tables(seqs) if cu_seqlens_k[-1] > cu_seqlens_q[-1] else None

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
