from dataclasses import dataclass
import torch
import torch.distributed as dist

class ParallelState:
    _instance_rank: int = 0
    _instance_world_size: int = 1

    @classmethod
    def set_info(cls, rank: int, world_size: int):
        cls._instance_rank = rank
        cls._instance_world_size = world_size

    @classmethod
    def get_rank(cls):
        return cls._instance_rank

    @classmethod
    def get_world_size(cls):
        return cls._instance_world_size

# 全局单例
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None    # 如果不是 None，则指明 kv cache存放地址，告诉 GPU 历史 KV 去哪读
    tp_group: dist.ProcessGroup | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, tp_group=None):
    global _CONTEXT
    current_tp_group = tp_group if tp_group is not None else _CONTEXT.tp_group
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, current_tp_group)

def reset_context():
    global _CONTEXT
    current_tp_group = _CONTEXT.tp_group
    _CONTEXT = Context(tp_group=current_tp_group)
