"""
Anthropic (Claude) LLM client.

Supported models:
- claude-opus-4-6             (high-end, most capable)
- claude-sonnet-4-6           (mid-tier, balanced)
- claude-haiku-4-5            (low-tier, fast and cheap)
"""

import os
import time
import anthropic

from clients.base import BaseLLMClient


class AnthropicClient(BaseLLMClient):

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages, **kwargs) -> str:
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Separate system message from conversation messages
        system_content = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        self._reset_last()

        api_kwargs = dict(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_content:
            api_kwargs["system"] = system_content

        start = time.perf_counter()
        response = self.client.messages.create(**api_kwargs)
        end = time.perf_counter()

        self.last_latency = end - start
        self.last_model = model
        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens
        self.last_total_tokens = response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text
