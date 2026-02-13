from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int | None = None,
        max_model_len: int | None = None,
        pd_separation: bool = False,
        **kwargs,
    ):
        # 修复：model 参数已作为位置参数传递，不应包含在 kwargs 字典中
        engine_args = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "pd_separation": pd_separation,
            **kwargs
        }
        
        self.llm_engine = LLMEngine(model, **engine_args)

    def generate(
        self,
        prompts: list[str] | list[list[int]] | str | list[int],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ):
        if isinstance(prompts, str) or (isinstance(prompts, list) and isinstance(prompts[0], int)):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        return self.llm_engine.generate(prompts, sampling_params, use_tqdm)

    def add_request(self, prompt, sampling_params):
        self.llm_engine.add_request(prompt, sampling_params)

    def step(self):
        return self.llm_engine.step()

    def is_finished(self):
        return self.llm_engine.is_finished()