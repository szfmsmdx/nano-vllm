import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory  

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.models.models import model_dict
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context, ParallelState
from nanovllm.utils.loader import load_model

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], shm_name: str = "nanovllm", ack_event: Event | list[Event] = None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank    # 全局rank
        self.event = event
        self.ack_event = ack_event 
        self.tensor_parallel_size = config.tensor_parallel_size
        self.instance_tp_size = config.instance_tp_size
        self.shm_name = shm_name

        # 初始化逻辑
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank, device_id=torch.device(f"cuda:{rank}"))
        torch.cuda.set_device(rank)

        # 建立通信组
        self.tp_group = None
        self.transfer_peer = None

        if config.pd_separation:
            prefill_ranks = list(range(0, self.instance_tp_size))
            decode_ranks = list(range(self.instance_tp_size, self.tensor_parallel_size))

            self.prefill_group = dist.new_group(prefill_ranks)
            self.decode_group = dist.new_group(decode_ranks)

            if rank in prefill_ranks:
                self.tp_group = self.prefill_group
                self.instance_rank = rank
                self.is_prefill_worker = True
                self.transfer_peer = rank + self.instance_tp_size
            else:
                self.tp_group = self.decode_group
                self.instance_rank = rank - self.instance_tp_size
                self.is_prefill_worker = False
                self.transfer_peer = rank - self.instance_tp_size
        else:
            self.tp_group = dist.group.WORLD
            self.instance_rank = rank
            self.is_prefill_worker = True

        ParallelState.set_info(self.instance_rank, self.instance_tp_size)
        set_context(False, tp_group=self.tp_group)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # self.model = Qwen3ForCausalLM(hf_config)
        # architectures = getattr(hf_config, "architectures", [])
        # if "Qwen2MoeForCausalLM" in architectures:
        #     self.model = Qwen3MoeForCausalLM(hf_config)
        # else:
        #     self.model = Qwen3ForCausalLM(hf_config)
        self.model = model_dict[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # 新增：传输专用流
        self.transfer_stream = torch.cuda.Stream()
        # 新增：保存异步通信句柄，防止被GC
        self.async_requests = []

        self.warmup_model()
        self.allocate_kv_cache()
        
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=self.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.shm_name)
                self.loop()

    def transfer_kv_cache(self, block_ids: list[int]):
        """
        使用 CUDA Stream + Async NCCL 实现计算与传输重叠
        """
        if not self.config.pd_separation or not block_ids:
            return 
        
        # 清理已完成的请求
        self.async_requests = [req for req in self.async_requests if not req.is_completed()]

        # 在专用传输流上执行，不阻塞默认计算流
        with torch.cuda.stream(self.transfer_stream):
            all_block_idx = torch.tensor(block_ids, device="cuda", dtype=torch.int64)
            num_blocks = all_block_idx.size(0)

            if self.is_prefill_worker:
                # 1. 提取数据 (Copy Kernel)
                data_to_send = self.kv_cache.index_select(2, all_block_idx)
                
                # 2. 异步发送 (Comm) - 不阻塞 CPU
                req = dist.isend(data_to_send, dst=self.transfer_peer)
                self.async_requests.append(req)
                
            else:
                recv_shape = list(self.kv_cache.shape)
                recv_shape[2] = num_blocks
                buffer = torch.empty(recv_shape, dtype=self.kv_cache.dtype, device="cuda")

                # 1. 异步接收 (Comm)
                req = dist.irecv(buffer, src=self.transfer_peer)
                self.async_requests.append(req)
                
                # 2. 等待接收完成 (在 GPU Stream 层面等待，而不是 CPU)
                # 注意：对于 PyTorch < 2.0，irecv 可能需要 req.wait() 才能保证数据就绪
                # 这里我们强制同步一下这个流，确保 copy_ 写回是安全的
                # 在极致优化中，可以使用 record_event / wait_event 让计算流等待
                req.wait() 

                # 3. 写回 Cache (Copy Kernel)
                self.kv_cache.index_copy_(2, all_block_idx, buffer)

        # 这里的 trick 是：
        # 我们没有调用 torch.cuda.synchronize() (除了 req.wait() 局部阻塞)
        # 这意味着 CPU 会立即返回，主进程可以继续去调度下一个 step
        # 而 GPU 会在 transfer_stream 上慢慢跑传输，不影响默认流上的计算

    def recv_decode_res(self, seq_len: int):
        if self.config.pd_separation and self.rank == 0:
            res_buffer = torch.zeros(seq_len, dtype=torch.long, device="cuda")
            # 结果传输通常很小，可以使用同步 recv 作为屏障
            dist.recv(res_buffer, src=self.transfer_peer)
            return res_buffer.tolist()
        return []

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
            
            if self.ack_event:
                if isinstance(self.ack_event, list):
                    pass
                else:
                    self.ack_event.set()

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
        if self.world_size > 1 and self.rank == 0:
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
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.instance_tp_size 
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
    
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q, max_seqlen_k = 0, 0
        slot_mapping = []

        for seq in seqs:
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

            for i in range(start_pos, end_pos):
                b_idx, b_offset = i // self.block_size, i % self.block_size
                slot = seq.block_table[b_idx] * self.block_size + b_offset
                slot_mapping.append(slot)

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
        group_rank = dist.get_rank(group=self.tp_group) if self.tp_group else 0
        temperatures = self.prepare_sample(seqs) if group_rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if group_rank == 0 else None

        if self.config.pd_separation and not is_prefill and self.instance_rank == 0 and not self.is_prefill_worker:
            res_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
            dist.send(res_tensor, dst=0)

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