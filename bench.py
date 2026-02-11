import os
import time
import json
import numpy as np
from datetime import datetime
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import SequenceStatus

def main():
    seed(0)
    # --- 配置参数 ---
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    # 替换为你本地的模型路径
    path = os.path.expanduser("/data3/szf_hf/huggingface/model/Qwen2.5-0.5B")
    
    if not os.path.exists(path):
        # 兼容性路径 fallback
        path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print(f"正在初始化 LLM: {path}")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # --- 准备测试数据 ---
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) 
        for _ in range(num_seqs)
    ]

    print(f"开始基准测试: 总请求数 {num_seqs}")
    
    # --- 温热 (Warmup) ---
    print("正在进行 Warmup...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10), use_tqdm=False)

    # --- 开始核心测量 ---
    # 我们认为所有请求在 arrival_time 同时到达 (Batch 提交场景)
    arrival_time = time.perf_counter()
    for p, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(p, sp)

    # 统计容器
    ttfts = {}          # seq_id -> latency (s)
    decode_latencies = [] # 每个 decode step 的平均单 token 耗时
    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    
    # 跟踪尚未获得 TTFT 的请求
    pending_ttft = {seq.seq_id for seq in llm.scheduler.waiting} | {seq.seq_id for seq in llm.scheduler.running}

    total_start = time.perf_counter()
    
    # 驱动推理循环
    while not llm.is_finished():
        step_start = time.perf_counter()
        
        # 1. 调度 (Schedule)
        seqs, is_prefill = llm.scheduler.schedule()

        # [修正 1] 在 postprocess 之前统计本轮真实的 token 数
        # 必须在这里读，因为 postprocess 会删除 current_chunk_size
        if is_prefill:
            # 统计本轮所有 Chunk 的大小之和
            step_tokens = sum(getattr(seq, "current_chunk_size", 0) for seq in seqs)
            prefill_tokens += step_tokens
        else:
            # Decode 阶段，每个 seq 产出一个 token
            step_tokens = len(seqs)
            decode_tokens += step_tokens
        
        # 2. 执行 (Run)
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        
        # 3. 后处理 (Postprocess)
        # 注意：这里会更新 seq.num_cached_tokens 并移除 current_chunk_size
        llm.scheduler.postprocess(seqs, token_ids)
        
        step_end = time.perf_counter()
        step_latency = step_end - step_start
        
        # [修正 2] 修正 TTFT 统计逻辑
        if is_prefill:
            prefill_time += step_latency
            for seq in seqs:
                # 只有当 seq 在本轮被标记为 RUNNING (说明跑完了最后一个 chunk)
                # 且它还在 pending 列表中时，才记录 TTFT
                if seq.seq_id in pending_ttft and seq.status == SequenceStatus.RUNNING:
                    ttfts[seq.seq_id] = step_end - arrival_time
                    pending_ttft.remove(seq.seq_id)
        else:
            decode_time += step_latency
            if len(seqs) > 0:
                decode_latencies.append(step_latency / len(seqs))
            
            # 容错：防止极少数情况下请求直接进入 decode 而漏记 TTFT
            for seq in seqs:
                if seq.seq_id in pending_ttft:
                    ttfts[seq.seq_id] = step_end - arrival_time
                    pending_ttft.remove(seq.seq_id)

    total_end = time.perf_counter()
    total_duration = total_end - total_start

    # --- 数据汇总与计算 ---
    ttft_values = list(ttfts.values())
    
    avg_ttft = np.mean(ttft_values) if ttft_values else 0
    p99_ttft = np.percentile(ttft_values, 99) if ttft_values else 0
    
    avg_tpot = np.mean(decode_latencies) if decode_latencies else 0
    p99_tpot = np.percentile(decode_latencies, 99) if decode_latencies else 0

    result = {
        "timestamp": datetime.now().isoformat(),
        "model": path,
        "config": {
            "num_seqs": num_seqs,
            "max_input_len": max_input_len,
            "max_output_len": max_ouput_len,
            "tensor_parallel_size": llm.model_runner.world_size,
            "enforce_eager": llm.model_runner.enforce_eager,
            "kvcache_block_size": llm.scheduler.block_manager.block_size,
            "chunk_size": llm.scheduler.chunk_size
        },
        "metrics": {
            "total_time_s": round(total_duration, 4),
            "total_tokens_processed": prefill_tokens + decode_tokens,
            "total_tokens_generated": decode_tokens,
            "throughput_tok_s": round((prefill_tokens + decode_tokens) / total_duration, 2),
            "prefill_throughput_tok_s": round(prefill_tokens / prefill_time, 2) if prefill_time > 0 else 0,
            "decode_throughput_tok_s": round(decode_tokens / decode_time, 2) if decode_time > 0 else 0,
            "avg_ttft_ms": round(avg_ttft * 1000, 2),
            "p99_ttft_ms": round(p99_ttft * 1000, 2),
            "avg_tpot_ms": round(avg_tpot * 1000, 2),
            "p99_tpot_ms": round(p99_tpot * 1000, 2),
        }
    }

    # --- 存储 JSON ---
    os.makedirs("./benchmark", exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.json")
    filepath = os.path.join("./benchmark", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*50)
    print(f"测试完成！结果已保存至: {filepath}")
    print("-" * 50)
    print(f"chunk size: {llm.scheduler.chunk_size}")
    print(f"总吞吐量: {result['metrics']['throughput_tok_s']} tok/s")
    print(f"平均 TTFT: {result['metrics']['avg_ttft_ms']} ms")
    print(f"P99 TTFT: {result['metrics']['p99_ttft_ms']} ms")
    print(f"平均 TPOT: {result['metrics']['avg_tpot_ms']} ms")
    print("="*50)

if __name__ == "__main__":
    main()