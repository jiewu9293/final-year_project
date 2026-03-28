"""
Google (Gemini) LLM client using new google-genai SDK.

Supported models:
- gemini-3.1-pro-preview           (high-end, most capable)
- gemini-3-flash-preview           (mid-tier, balanced)
- gemini-3.1-flash-lite-preview    (low-tier, fast and cheap)
"""

import os
import time
from google import genai
from google.genai import types

from clients.base import BaseLLMClient


class GoogleClient(BaseLLMClient):

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)

    def generate(self, messages, **kwargs) -> str:
        model_name = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Convert OpenAI-style messages to Gemini format
        system_instruction = None
        contents = []

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        self._reset_last()
        start = time.perf_counter()

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        end = time.perf_counter()

        self.last_latency = end - start
        self.last_model = model_name

        # Extract token usage
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_token_count", None)
            self.last_output_tokens = getattr(usage, "candidates_token_count", None)
            total = (self.last_input_tokens or 0) + (self.last_output_tokens or 0)
            self.last_total_tokens = total if total > 0 else None

        return response.text
