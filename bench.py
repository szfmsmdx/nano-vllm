import os
import time
import json
import argparse
import numpy as np
from datetime import datetime
from random import randint, seed
from nanovllm import LLM, SamplingParams

# 设置环境变量，优化 CPU 计算
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

def get_args():
    parser = argparse.ArgumentParser(description="Nano-vLLM Benchmark")
    parser.add_argument("--model", type=str, default="/data3/szf_hf/huggingface/model/Qwen3-0.6B", help="Model path")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Max input length")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Max output length")
    parser.add_argument("--tp", type=int, default=4, help="Tensor Parallel Size (Total GPUs)")
    parser.add_argument("--pd", action="store_true", help="Enable Prefill-Decode Separation")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager execution")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size for prefill")
    return parser.parse_args()

def main():
    args = get_args()
    seed(0)

    # 路径检查
    path = os.path.expanduser(args.model)
    if not os.path.exists(path):
        alt_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
        if os.path.exists(alt_path):
            print(f"Warning: Model not found at {path}, using {alt_path}")
            path = alt_path

    print(f"\n{'='*50}")
    print(f"Benchmark Configuration:")
    print(f"  Model: {path}")
    print(f"  TP Size: {args.tp}")
    print(f"  PD Separation: {args.pd}")
    print(f"  Num Seqs: {args.num_seqs}")
    print(f"  Chunk Size: {args.chunk_size}")
    print(f"{'='*50}\n")

    # 初始化 LLM
    llm = LLM(
        path, 
        enforce_eager=args.enforce_eager, 
        max_model_len=4096, 
        tensor_parallel_size=args.tp, 
        pd_separation=args.pd,
        chunk_size=args.chunk_size,
        gpu_memory_utilization=0.5
    )

    # --- 准备测试数据 ---
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))] for _ in range(args.num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len)) 
        for _ in range(args.num_seqs)
    ]

    print(f"Starting warmup...")
    llm.generate(["Warmup"], SamplingParams(max_tokens=10), use_tqdm=False)

    print(f"Starting benchmark...")
    arrival_time = time.perf_counter()
    for p, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(p, sp)

    ttfts = {}          
    decode_latencies = [] 
    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0

    total_start = time.perf_counter()
    
    while not llm.is_finished():
        step_start = time.perf_counter()
        
        # 调用 step，解包返回值
        scheduled_seqs, is_prefill, num_step_tokens = llm.step()
        
        step_end = time.perf_counter()
        step_latency = step_end - step_start
        
        if not scheduled_seqs:
            continue

        # 统计吞吐量
        if is_prefill:
            prefill_time += step_latency
            prefill_tokens += num_step_tokens
        else:
            decode_time += step_latency
            actual_decode_tokens = -num_step_tokens
            decode_tokens += actual_decode_tokens
            if len(scheduled_seqs) > 0:
                decode_latencies.append(step_latency / len(scheduled_seqs))

        # 统计 TTFT
        for seq in scheduled_seqs:
            if seq.seq_id not in ttfts:
                if seq.num_completion_tokens > 0 or not is_prefill:
                    ttfts[seq.seq_id] = step_end - arrival_time

    total_end = time.perf_counter()
    total_duration = total_end - total_start

    # --- 数据汇总 ---
    ttft_values = list(ttfts.values())
    avg_ttft = np.mean(ttft_values) if ttft_values else 0
    p99_ttft = np.percentile(ttft_values, 99) if ttft_values else 0
    avg_tpot = np.mean(decode_latencies) if decode_latencies else 0
    p99_tpot = np.percentile(decode_latencies, 99) if decode_latencies else 0

    # 按照之前的格式构造结果
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": path,
        "config": {
            "num_seqs": args.num_seqs,
            "max_input_len": args.max_input_len,
            "max_output_len": args.max_output_len,
            "tensor_parallel_size": args.tp,
            "enforce_eager": args.enforce_eager,
            "kvcache_block_size": llm.config.kvcache_block_size,
            "chunk_size": args.chunk_size,
            "pd_separation": args.pd
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

    os.makedirs("./benchmark", exist_ok=True)
    filename = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pd{args.pd}.json"
    filepath = os.path.join("./benchmark", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"TP: {args.tp} | Chunk Size: {args.chunk_size} | Use PD: {args.pd}")
    print(f"Results saved to: {filepath}")
    print(f"Total Throughput: {result['metrics']['throughput_tok_s']} tok/s")
    print(f"Prefill Throughput: {result['metrics']['prefill_throughput_tok_s']} tok/s")
    print(f"Decode Throughput: {result['metrics']['decode_throughput_tok_s']} tok/s")
    print(f"Avg TTFT: {result['metrics']['avg_ttft_ms']} ms")
    print(f"Avg TPOT: {result['metrics']['avg_tpot_ms']} ms")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()