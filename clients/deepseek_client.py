"""
DeepSeek LLM client (OpenAI-compatible API).

Supported models:
- deepseek-chat      (DeepSeek-V3, general purpose)
- deepseek-reasoner  (DeepSeek-R1, chain-of-thought reasoning)
"""

import os
import time
from openai import OpenAI

from clients.base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):

    BASE_URL = "https://api.deepseek.com"

    # Reasoning models do not support custom temperature
    REASONING_MODELS = {"deepseek-reasoner"}

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def generate(self, messages, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        api_kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )

        # deepseek-reasoner does not support temperature parameter
        if model not in self.REASONING_MODELS:
            api_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

        self._reset_last()

        start = time.perf_counter()
        response = self.client.chat.completions.create(**api_kwargs)
        end = time.perf_counter()

        self.last_latency = end - start
        self.last_model = getattr(response, "model", None) or model

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_tokens", None)
            self.last_output_tokens = getattr(usage, "completion_tokens", None)
            self.last_total_tokens = getattr(usage, "total_tokens", None)

        return response.choices[0].message.content
