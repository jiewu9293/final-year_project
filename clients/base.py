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
        self.last_energy_kwh: Tuple[float, float] | None = None
        self.last_gwp_kgco2eq: Tuple[float, float] | None = None
        self.last_adpe_kgsbeq: Tuple[float, float] | None = None
        self.last_pe_mj: Tuple[float, float] | None = None

    def _reset_last(self):
        """Reset all per-call tracking attributes before a new API call."""
        self.last_latency = None
        self.last_input_tokens = None
        self.last_output_tokens = None
        self.last_total_tokens = None
        self.last_model = None
        self.last_energy_kwh = None
        self.last_gwp_kgco2eq = None
        self.last_adpe_kgsbeq = None
        self.last_pe_mj = None

    def _extract_impacts(self, response):
        """Extract EcoLogits environmental impact data from the response."""
        impacts = getattr(response, "impacts", None)
        if impacts is not None:
            def _to_range(val):
                if hasattr(val, 'min'):
                    return float(val.min), float(val.max)
                return float(val), float(val)
            self.last_energy_kwh = _to_range(impacts.energy.value)
            self.last_gwp_kgco2eq = _to_range(impacts.gwp.value)
            self.last_adpe_kgsbeq = _to_range(impacts.adpe.value)
            self.last_pe_mj = _to_range(impacts.pe.value)

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """ Generate a response given a prompt. """
        raise NotImplementedError