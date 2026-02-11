import os
import time
import json
import numpy as np
from datetime import datetime
from random import randint, seed
from nanovllm import LLM, SamplingParams

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
    # prefill 完就放到 running了，所以可能会有 seq 没有decode但是也在 running 的情况
    pending_ttft = {seq.seq_id for seq in llm.scheduler.waiting} | {seq.seq_id for seq in llm.scheduler.running}

    total_start = time.perf_counter()
    
    # 驱动推理循环
    while not llm.is_finished():
        step_start = time.perf_counter()
        
        # 为了获取颗粒度指标，我们手动执行 step() 内部逻辑
        seqs, is_prefill = llm.scheduler.schedule()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids)
        
        step_end = time.perf_counter()
        step_latency = step_end - step_start
        
        if is_prefill:
            prefill_time += step_latency
            # 计算本轮 prefill 处理的 token 总数 (prompt - cached)
            batch_prefill_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs)
            prefill_tokens += batch_prefill_tokens
            
            # 在当前 nano-vllm 实现中，prefill step 完成即意味着首字产生
            for seq in seqs:
                if seq.seq_id in pending_ttft:
                    ttfts[seq.seq_id] = step_end - arrival_time
                    pending_ttft.remove(seq.seq_id)
        else:
            decode_time += step_latency
            decode_tokens += len(seqs) # Decode 阶段每人产出一个 token
            # TPOT = 这一步的总耗时 / 步内并发请求数
            if len(seqs) > 0:
                decode_latencies.append(step_latency / len(seqs))
            
            # 容错：防止某些请求直接跳过 prefill 进入 decode
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
    print(f"总吞吐量: {result['metrics']['throughput_tok_s']} tok/s")
    print(f"平均 TTFT: {result['metrics']['avg_ttft_ms']} ms")
    print(f"P99 TTFT: {result['metrics']['p99_ttft_ms']} ms")
    print(f"平均 TPOT: {result['metrics']['avg_tpot_ms']} ms")
    print("="*50)

if __name__ == "__main__":
    main()