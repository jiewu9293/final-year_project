# clients/openai_client.py

import os
import time
from openai import OpenAI
from openai import NotFoundError
from ecologits import EcoLogits

from clients.base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """
    OpenAI LLM client.

    Supported models:
    - gpt-5.4              (high-end, most capable)
    - gpt-5.4-mini         (mid-tier, balanced)
    - gpt-5.4-nano         (low-tier, fast and cheap)
    """

    # Reasoning models do not support custom temperature
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini"}

    def __init__(
        self,
        model: str = "gpt-5.4",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment variables"
            )

        EcoLogits.init(providers="openai")
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages, **kwargs) -> str:
        """
        messages: list[dict] like [{"role":"user","content":"..."}]
        kwargs may include: model, temperature, max_tokens
        """
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        api_kwargs = dict(
            model=model,
            messages=messages,
        )

        # gpt-5.x series uses max_completion_tokens instead of max_tokens
        if model.startswith("gpt-5"):
            api_kwargs["max_completion_tokens"] = max_tokens
        else:
            api_kwargs["max_tokens"] = max_tokens

        # o1/o3 reasoning models do not support custom temperature
        if model not in self.REASONING_MODELS:
            api_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

        self._reset_last()

        try:
            start = time.perf_counter()
            response = self.client.chat.completions.create(**api_kwargs)
            end = time.perf_counter()
        except NotFoundError as e:
            raise NotFoundError()

        self.last_latency = end - start
        self.last_model = getattr(response, "model", None) or model

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_tokens", None)
            self.last_output_tokens = getattr(usage, "completion_tokens", None)
            self.last_total_tokens = getattr(usage, "total_tokens", None)

        # Try to extract environmental impacts (may not be available for all models)
        try:
            self._extract_impacts(response)
        except (AttributeError, TypeError):
            pass  # Impacts not available for this model

        return response.choices[0].message.content

