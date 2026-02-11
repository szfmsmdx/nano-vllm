from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最长 decode 长度
        self.max_num_batched_tokens = config.max_num_batched_tokens # prefill 塞进去的最大长度
        self.chunk_size = config.chunk_size
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
        # 加入 chunk prefill 特性
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # prefill
        if self.waiting:
            self.waiting = sorted(self.waiting, key=lambda s: len(s) - s.num_cached_tokens)
            finished_prefills = []
            for seq in self.waiting:
                if num_seqs >= self.max_num_seqs:
                    break

                total_buget_remain = self.max_num_batched_tokens - num_batched_tokens
                if total_buget_remain <= 0:
                    break

                if seq.num_cached_tokens == 0:
                    if not self.block_manager.can_allocate(seq):
                        break
                    self.block_manager.allocate(seq)

                remain_len = len(seq) - seq.num_cached_tokens
                chunk_len = min(remain_len, self.chunk_size, total_buget_remain)

                if chunk_len <= 0:
                    continue

                seq.current_chunk_size = chunk_len
                num_batched_tokens += chunk_len
                num_seqs += 1
                scheduled_seqs.append(seq)

                if seq.num_cached_tokens + chunk_len == len(seq):
                    seq.status = SequenceStatus.RUNNING
                    finished_prefills.append(seq)

            if scheduled_seqs:
                for seq in finished_prefills:
                    self.waiting.remove(seq)
                    self.running.append(seq)
                return scheduled_seqs, True
            
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break

            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
            
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False
        
        return [], False
            

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            chunk_size = getattr(seq, "current_chunk_size", 0)
            if chunk_size > 0:  # 说明是 prefill 阶段
                seq.num_cached_tokens += chunk_size
                delattr(seq, "current_chunk_size")
                if seq.num_cached_tokens == len(seq):
                    if token_id is not None:
                        seq.append_token(token_id)
                    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        if seq in self.running:
                            self.running.remove(seq)
            else:   # decode 阶段
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running: self.running.remove(seq)