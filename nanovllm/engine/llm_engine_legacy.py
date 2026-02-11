import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

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
        self.ps = []
        self.events = []
        # CUDA 强制要求 -> spawn 启动方法会创建完全干净的新py解释器，确保每个进程能够独立、安全初始化CUDA环境
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event() # 子进程同步事件对象
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建子进程，target是进程启动后的类或函数
            process.start() # 启动子进程，启动完就不管了
            
            self.ps.append(process) # 方便后续的进程管理
            self.events.append(event)   # 用于主进程和子进程间的通信
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # 注册 exit，不管崩溃还是正常结束都会调用 exit

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行引擎的一个推理步 (Step)。
        
        这是推理循环的核心函数，每一次调用都会推动系统向前走一步：
        1. 调度 (Schedule): 决定当前是做 Prefill 还是 Decode，选出要运行的 Sequence。
        2. 执行 (Run): 调用 ModelRunner (可能在另一个进程) 执行模型的前向传播和采样。
        3. 后处理 (Postprocess): 更新 Sequence 的状态 (追加 Token、检查结束条件、释放资源)。
        
        Returns:
            tuple:
                - outputs (list): 本轮刚刚完成 (Finished) 的请求结果列表。
                - num_tokens (int): 本轮处理的 Token 总数 (用于计算吞吐量)。
                                    如果是 Prefill，为总 Input Token 数；
                                    如果是 Decode，为生成的 Token 数 (即 Batch Size, 负数表示)。
        """
        # 1. 调度阶段
        # 询问 Scheduler 本轮该跑哪些请求 (seqs)，以及是 Prefill 还是 Decode 模式 (is_prefill)
        seqs, is_prefill = self.scheduler.schedule()

        # 2. 模型执行阶段
        # 通过 IPC (跨进程通信) 调用 ModelRunner 的 run 方法
        # 输入: 本轮调度的 seqs 和模式标志
        # 输出: 采样生成的 token_ids 列表 (每个 seq 对应一个新 token)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 后处理阶段
        # 将生成的 token_ids 追加到对应的 Sequence 中
        # 检查是否遇到 EOS 或达到最大长度，如果完成则释放 Block 资源
        self.scheduler.postprocess(seqs, token_ids)

        # 4. 结果收集与统计
        # 收集本轮所有变为 FINISHED 状态的请求，准备返回给用户
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 计算本轮吞吐量统计指标
        # 如果是 Prefill，处理的是 prompt 中的所有 token (sum(len(seq)))
        # 如果是 Decode，处理的是并行生成的 token 数 (即 -len(seqs), 这里的负号是内部约定的标记，用于由外层区分阶段)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

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
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
