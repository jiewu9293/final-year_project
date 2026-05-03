from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMResponse:
    """
    Standardized response returned by all LLM clients.
    """
    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None

class BaseLLMClient(ABC):
    """ Abstract base class for all LLM clients. """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.last_latency: float | None = None
        self.last_input_tokens: int | None = None
        self.last_output_tokens: int | None = None
        self.last_total_tokens: int | None = None
        self.last_model: str | None = None

    def _reset_last(self):
        """Reset all per-call tracking attributes before a new API call."""
        self.last_latency = None
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        self.last_model = None

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """ Generate a response given a prompt. """
        raise NotImplementedError