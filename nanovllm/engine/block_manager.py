from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id    # 物理id
        self.ref_count = 0          # 引用计数（共享前缀）
        self.hash = -1              # 块存储的token哈希
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]    # 所有 block
        self.hash_to_block_id: dict[int, int] = dict()  # 记录那些Token序列放在那个物理块里
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 需要取用，所以用deque设计
        self.used_block_ids: set[int] = set()   # 知道有没有被用就行

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0 # 当前有多少个seq在使用这个 block
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks   # 分配不出 block ids了

    def allocate(self, seq: Sequence):
        """
        对 prefill sequence请求分配block
        """
        assert not seq.block_table  # 条件为 false 时执行，如果为空就是 true，就不用 manager 分配了
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks): # 为每个块分配
            token_ids = seq.block(i)    # 取出这个块的 token_ids
            # 取出的 token ids 和 block size相等，说明这个 block 是满的，这时候计算prefix caching，可以被 hash
            # 这里 h 也作为参数，所以要求 prefix cache 是连续相等才能用
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 看看之前别人有没有分配过
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # 如果没存过，或者存储的 tokenids 对不上（防止哈希碰撞）
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]   # 从 free block 中拿一个出来
                block = self._allocate_block(block_id)
            else:   # cache命中
                seq.num_cached_tokens += self.block_size    # prefill 阶段多了 block size个词已经cache过了
                if block_id in self.used_block_ids:         
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)  # cache 命中但是不在使用队列，就肯定在 free 队列且ref_count=0
            if h != -1: # 如果block 满会分配h，否则不分配就是 -1就跳过
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
