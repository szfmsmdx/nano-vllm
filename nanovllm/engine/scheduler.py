from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunk_size = config.chunk_size
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() 
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        prefill_seqs = []
        decode_seqs = []
        
        # --- 1. 调度 Decode ---
        # 优先处理 Running 队列中的任务
        current_running = list(self.running)
        self.running.clear()
        
        num_decode_seqs = 0
        for seq in current_running:
            if num_decode_seqs >= self.max_num_seqs:
                self.preempt(seq)
                continue
                
            while not self.block_manager.can_append(seq):
                if decode_seqs:
                    preempted_seq = decode_seqs.pop()
                    self.preempt(preempted_seq)
                    num_decode_seqs -= 1
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                num_decode_seqs += 1
        
        self.running.extend(decode_seqs)

        # --- 2. 调度 Prefill ---
        if self.waiting:
            # 按剩余长度排序（短任务优先或FIFO，这里保持原逻辑）
            sorted_waiting = sorted(self.waiting, key=lambda s: len(s) - s.num_cached_tokens)
            self.waiting = deque(sorted_waiting)
            
            num_batched_tokens = 0
            seqs_to_remove = [] # 仅移除那些在本轮彻底完成 Prefill 的任务
            
            for seq in self.waiting:
                if num_decode_seqs + len(prefill_seqs) >= self.max_num_seqs:
                    break

                total_budget_remain = self.max_num_batched_tokens - num_batched_tokens
                if total_budget_remain <= 0:
                    break

                # 首次调度分配 Block
                if seq.num_cached_tokens == 0:
                    if not self.block_manager.can_allocate(seq):
                        break
                    self.block_manager.allocate(seq)

                remain_len = len(seq) - seq.num_cached_tokens
                chunk_len = min(remain_len, self.chunk_size, total_budget_remain)

                if chunk_len <= 0:
                    continue

                seq.current_chunk_size = chunk_len
                num_batched_tokens += chunk_len
                prefill_seqs.append(seq)

                # 只有当本轮跑完就彻底结束 Prefill 时，才标记移除
                if seq.num_cached_tokens + chunk_len == len(seq):
                    seqs_to_remove.append(seq)

            # 统一移除已完成的任务
            for seq in seqs_to_remove:
                self.waiting.remove(seq)
        
        return prefill_seqs, decode_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            chunk_size = getattr(seq, "current_chunk_size", 0)
            
            if chunk_size > 0:  # Prefill 阶段
                seq.num_cached_tokens += chunk_size
                delattr(seq, "current_chunk_size")
                
                # 如果 Prefill 全部完成
                if seq.num_cached_tokens == len(seq):
                    if token_id is not None:
                        seq.append_token(token_id)
                    
                    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                    else:
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                # 如果未完成 (Chunked)，它仍然在 waiting 队列中，等待下一轮调度
                # 状态保持 WAITING，num_cached_tokens 已更新
            
            else:   # Decode 阶段
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running: 
                        self.running.remove(seq)