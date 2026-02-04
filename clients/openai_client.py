# clients/openai_client.py

import os
import time
from openai import OpenAI
from openai import NotFoundError

from clients.base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """
    OpenAI LLM client.
    Responsible ONLY for:
    - calling OpenAI API
    - measuring latency
    - returning a standardized LLMResponse
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
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

        self.client = OpenAI(api_key=api_key)

        self.last_latency: float | None = None
        self.last_input_tokens: int | None = None
        self.last_output_tokens: int | None = None
        self.last_total_tokens: int | None = None
        self.last_model: str | None = None

    def generate(self, messages, **kwargs) -> str:
        """
        messages: list[dict] like [{"role":"user","content":"..."}]
        kwargs may include: model, temperature, max_tokens
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        self.last_latency = None
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        self.last_model = None

        try:
            start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            end = time.perf_counter()
        except NotFoundError as e:
            raise NotFoundError(
                message=(
                    f"Model '{model}' was not found or you do not have access to it. "
                    "Set OPENAI_MODEL to a model available to your account, or update the 'model' argument."
                ),
                response=e.response,
                body=e.body,
            ) from e

        self.last_latency = end - start
        self.last_model = getattr(response, "model", None) or model

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_tokens", None)
            self.last_output_tokens = getattr(usage, "completion_tokens", None)
            self.last_total_tokens = getattr(usage, "total_tokens", None)

        return response.choices[0].message.content

