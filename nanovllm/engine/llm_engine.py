import atexit
import pickle
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        
        ctx = mp.get_context("spawn")
        
        if not config.pd_separation:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)

        else:
            prefill_events = []
            for i in range(1, config.instance_tp_size):
                event = ctx.Event()
                # 预填充worker不需要额外的ACK，因为它们是同步的
                process = ctx.Process(target=ModelRunner, args=(config, i, event, "nanovllm_prefill"))
                process.start()
                self.ps.append(process)
                prefill_events.append(event)


            decode_events = []
            decode_ack_events = [] # 新增: ACK 事件列表
            self.shm_decode = SharedMemory(name="nanovllm_decode", create=True, size=2**20)

            for i in range(config.instance_tp_size, config.tensor_parallel_size):
                event = ctx.Event()
                ack_event = ctx.Event() # 为每个Decode Worker创建ACK
                # 将 ack_event 传入子进程
                process = ctx.Process(target=ModelRunner, args=(config, i, event, "nanovllm_decode", ack_event))
                process.start()
                self.ps.append(process)
                decode_events.append(event)
                decode_ack_events.append(ack_event)
            
            self.decode_events = decode_events
            self.decode_ack_events = decode_ack_events
            self.prefill_runner = ModelRunner(config, 0, prefill_events, "nanovllm_prefill")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def _drive_decode_runner(self, method_name, *args):
        """
        发送命令给 Decode Workers，并等待它们确认接收 (ACK)。
        这样可以防止下一条命令覆盖未被读取的命令。
        """
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm_decode.buf[0:4] = n.to_bytes(4, "little")
        self.shm_decode.buf[4:n+4] = data
        
        # 1. 触发子进程读取
        for e in self.decode_events:
            e.set()
        
        # 2. 【关键】等待所有子进程确认读取完毕
        for e in self.decode_ack_events:
            e.wait()
            e.clear() # 清除标志，为下一次做准备

    def exit(self):
        if not self.config.pd_separation:
            self.model_runner.call("exit")
            del self.model_runner
        else:
            self._drive_decode_runner("exit")
            self.prefill_runner.call("exit")
            self.shm_decode.close()
            self.shm_decode.unlink()
            del self.prefill_runner

        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 1. 调度
        if not self.config.pd_separation:
            prefill_seqs, decode_seqs = self.scheduler.schedule()
            is_prefill = len(prefill_seqs) > 0
            seqs = prefill_seqs if is_prefill else decode_seqs
            if not seqs: return [], 0, 0
            
            # 先统计
            prefill_count = sum(getattr(seq, "current_chunk_size", 0) for seq in seqs) if is_prefill else 0
            decode_count = len(seqs) if not is_prefill else 0

            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            
            return seqs, prefill_count, decode_count
        
        else:
            prefill_seqs, decode_seqs = self.scheduler.schedule()
            
            if not prefill_seqs and not decode_seqs:
                return [], 0, 0

            # 2. 统计
            prefill_count = sum(getattr(seq, "current_chunk_size", 0) for seq in prefill_seqs)
            decode_count = len(decode_seqs)

            # 3. 异步Decode
            if decode_seqs:
                self._drive_decode_runner("run", decode_seqs, False)

            # 4. 同步Prefill
            prefill_token_ids = []
            if prefill_seqs:
                prefill_token_ids = self.prefill_runner.call("run", prefill_seqs, True)

            # 5. 收集
            decode_token_ids = []
            if decode_seqs:
                decode_token_ids = self.prefill_runner.recv_decode_res(len(decode_seqs))

            # 6. KV Cache 传输
            blocks_to_transfer = set()
            for seq in prefill_seqs:
                if seq.num_cached_tokens + getattr(seq, "current_chunk_size", 0) == len(seq):
                    blocks_to_transfer.update(seq.block_table)
            
            if blocks_to_transfer:
                block_list = list(blocks_to_transfer)
                self._drive_decode_runner("transfer_kv_cache", block_list)
                self.prefill_runner.call("transfer_kv_cache", block_list)

            # 7. 后处理
            self.scheduler.postprocess(prefill_seqs, prefill_token_ids)
            self.scheduler.postprocess(decode_seqs, decode_token_ids)

            return prefill_seqs + decode_seqs, prefill_count, decode_count
    
    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | str,
        sampling_params: SamplingParams | list[SamplingParams] | None=None,
        use_tqdm: bool = True,
    ) -> list[str]:
        if isinstance(prompts, str) or (isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int)):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = 0.
        decode_throughput = 0.
        alpha = 0.1

        while not self.is_finished():
            t = perf_counter()
            seqs, prefill_count, decode_count = self.step()
            dt = perf_counter() - t
            
            if use_tqdm and dt > 0:
                if prefill_count > 0:
                    current_tp = prefill_count / dt
                    prefill_throughput = prefill_throughput * (1 - alpha) + current_tp * alpha
                if decode_count > 0:
                    current_tp = decode_count / dt
                    decode_throughput = decode_throughput * (1 - alpha) + current_tp * alpha
                    
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            current_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            for seq_id, token_ids in current_outputs:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs