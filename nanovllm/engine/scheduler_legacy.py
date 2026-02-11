from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最长 decode 长度
        self.max_num_batched_tokens = config.max_num_batched_tokens # prefill 塞进去的最大长度
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # continues batching基础实现
        self.waiting: deque[Sequence] = deque() 
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度器的主函数，执行 Continuous Batching 调度策略。
        
        逻辑流程：
        1. 优先检查等待队列 (Waiting Queue)，执行 Prefill (预填充) 调度。
           如果显存足够，将等待中的请求分配显存并移入运行队列。
        2. 如果本轮有 Prefill 任务，直接返回 (Prefill 阶段不与 Decode 混合执行)。
        3. 如果没有 Prefill 任务，检查运行队列 (Running Queue)，执行 Decode (解码) 调度。
           如果显存不足以容纳新 Token，触发抢占机制 (Preemption) 释放资源。
        
        Returns:
            tuple[list[Sequence], bool]:
                - scheduled_seqs: 本轮决定执行的 Sequence 列表。
                - is_prefill: 本轮是否为 Prefill 阶段 (True: Prefill, False: Decode)。
        """
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # 遍历等待队列，只要当前批次 sequence 数量未达上限，就尝试加入新的 Prefill 请求
        while self.waiting and num_seqs < self.max_num_seqs:    # prefill 优先调度
            seq = self.waiting[0]   # 查看队首请求 (暂不移出队列，需先判断能否分配)
            # num_batched_tokens : 本轮积攒的 prefill Token 长度
            # len(seq) 当前请求 seq 的长度
            # max_num_batched_tokens 一次性最多多少个
            
            # 资源检查：1. 加入该请求后是否超过最大 token 批处理限制；2. BlockManager 是否有足够空闲块
            # prefill 可以用 len 来判断
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break   # 资源不足，停止调度后续等待请求
            
            num_seqs += 1
            self.block_manager.allocate(seq)    # 真正分配物理显存块 (Block)
            # 这个 seq 真正要算的部分
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 累加本轮计算负载 (扣除命中 Prefix Cache 的部分)
            seq.status = SequenceStatus.RUNNING # 更新状态为运行中
            self.waiting.popleft()  # 取出 prefill 的 (从等待队列移除)
            self.running.append(seq)    # 加入运行队列尾部
            scheduled_seqs.append(seq)  # 加入本轮调度名单
            
        # 如果本轮调度了 Prefill 任务，则立即返回，优先执行 Prefill (与 Decode 分离)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 进入 Decode 阶段：遍历运行队列，为每个 Sequence 分配下一个 Token 所需的资源
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()    # 取出队首的 Sequence 进行调度检查
            
            # 循环检查：当前 seq 是否能分配到追加 Token 所需的新 Block
            while not self.block_manager.can_append(seq):
                # 显存不足 (OOM)，触发抢占逻辑 (Preemption) 以释放空间
                if self.running:
                    # 牺牲策略：优先抢占运行队列中最晚加入的任务 (队尾)，将其踢回等待队列并释放显存
                    self.preempt(self.running.pop())
                else:
                    # 如果队列里没有其他任务可抢，只能抢占当前任务自己
                    self.preempt(seq)
                    break   # 自己被抢占了，停止处理该 seq
            else:
                # Python 的 while-else 语法：仅当 while 循环条件为 False (即 can_append 成功) 且未触发 break 时执行
                # 说明资源足够，没有发生针对当前 seq 的抢占
                num_seqs += 1
                self.block_manager.may_append(seq)  # 如果跨越了 Block 边界，分配新的物理块
                scheduled_seqs.append(seq)          # 加入本轮调度名单
        
        # 断言保护：理论上 decode 阶段进入循环后应该至少能调度一个任务，或者触发抢占逻辑
        assert scheduled_seqs
        # 将本轮成功调度的任务放回 running 队列头部 (保持 Round-Robin 顺序，供下一轮使用)
        # 使用 reversed 是为了抵消 popleft 和 extendleft 造成的顺序反转，保持原序
        self.running.extendleft(reversed(scheduled_seqs))
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 判断结束状态
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
