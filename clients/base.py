from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """ Generate a response given a prompt. """
        raise NotImplementedError