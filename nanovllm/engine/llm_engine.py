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
        # CUDA 强制要求 -> spawn 启动方法会创建完全干净的新py解释器，确保每个进程能够独立、安全初始化CUDA环境
        ctx = mp.get_context("spawn")
        
        if not config.pd_separation:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event() # 子进程同步事件对象
                process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建子进程，target是进程启动后的类或函数
                process.start() # 启动子进程，启动完就不管了
                
                self.ps.append(process) # 方便后续的进程管理
                self.events.append(event)   # 用于主进程和子进程间的通信
            self.model_runner = ModelRunner(config, 0, self.events)

        else:
            prefill_events = []
            for i in range(1, config.instance_tp_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event, "nanovllm_prefill"))
                process.start()
                self.ps.append(process)
                prefill_events.append(event)


            decode_events = []
            self.shm_decode = SharedMemory(name="nanovllm_decode", create=True, size=2**20)

            for i in range(config.instance_tp_size, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event, "nanovllm_decode"))
                process.start()
                self.ps.append(process)
                decode_events.append(event)
            
            self.decode_events = decode_events
            self.prefill_runner = ModelRunner(config, 0, prefill_events, "nanovllm_prefill")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # 注册 exit，不管崩溃还是正常结束都会调用 exit

    def _drive_decode_runner(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm_decode.buf[0:4] = n.to_bytes(4, "little")
        self.shm_decode.buf[4:n+4] = data
        for e in self.decode_events:
            e.set()

    def exit(self):
        if not self.config.pd_separation:
            self.model_runner.call("exit")
            del self.model_runner
        else:
            self.prefill_runner.call("exit")
            self._drive_decode_runner("exit")
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
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs: return [], False, 0
        num_tokens = 0

        if not self.config.pd_separation:
            num_tokens = sum(getattr(seq, "current_chunk_size", 0) for seq in seqs) if is_prefill else -len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            # outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        else:
            if is_prefill:
                num_tokens = sum(getattr(seq, "current_chunk_size", 0) for seq in seqs)
                token_ids = self.prefill_runner.call("run", seqs, True)
                blocks_to_transfer = set()
                for seq in seqs:
                    blocks_to_transfer.update(seq.block_table)
                block_list = list(blocks_to_transfer)

                self._drive_decode_runner("transfer_kv_cache", block_list)
                self.prefill_runner.call("transfer_kv_cache", block_list)
                self.scheduler.postprocess(seqs, token_ids)
            else:
                num_tokens = -len(seqs)
                self._drive_decode_runner("run", seqs, False)
                token_ids = self.prefill_runner.recv_decode_res(len(seqs))
                self.scheduler.postprocess(seqs, token_ids)
                # outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return seqs, is_prefill, num_tokens
    
    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()  # 记录时间的
            seqs, is_prefill, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
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
