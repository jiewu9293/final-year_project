from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Task:
    """
    Benchmark task abstraction (minimal).
    You can extend fields to match UnLeakedTestBench task schema.
    """
    task_id: str
    language: str  # e.g., "python"
    source_code: str  # function / module under test
    prompt_text: str  # task description / docstring / instructions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """
    Provider-agnostic chat message.
    Most LLM SDKs accept a list of {role, content}.
    """
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class PromptBundle:
    """
    A prompt that will be sent to an LLM.
    - messages: chat format (preferred)
    - extra: any extra bookkeeping you want (e.g. few-shot examples used)
    - trace: ToT/GToT can append intermediate steps here for logging
    """
    framework: str
    task_id: str
    messages: List[Message]
    extra: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list) #design for tot/gtot

    def to_openai_like(self) -> List[Dict[str, str]]:
        """Convenience: convert to common SDK message format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


class LLMClient(Protocol):
    """
    A minimal client protocol your 'clients/' implementations should satisfy.
    You can wrap OpenAI/Anthropic/Gemini/DeepSeek behind this interface.
    """
    def generate(self, messages: Sequence[Dict[str, str]], **kwargs) -> str:
        ...


class TemplateRenderer:
    """
    Lightweight template renderer:
    - Loads .txt templates from prompting/templates/
    - Supports {placeholders} using Python .format(**vars)
    """
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def load(self, filename: str) -> str:
        path = self.templates_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, filename: str, **vars: Any) -> str:
        raw = self.load(filename)
        try:
            return raw.format(**vars)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Missing template variable '{missing}' for template '{filename}'. "
                f"Provided keys: {sorted(vars.keys())}"
            ) from e


class PromptFramework(ABC):
    """
    Base class for prompt frameworks (Zero-shot / Few-shot / CoT / ToT / GToT).

    Design goals:
    - Each framework knows how to turn a Task into a PromptBundle (messages).
    - ToT/GToT can be multi-step: base provides optional step() and trace.
    - Provider clients can consume PromptBundle.messages directly.
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ):
        self.templates_dir = templates_dir or (Path(__file__).parent / "templates")
        self.renderer = TemplateRenderer(self.templates_dir)
        self.system_prompt = system_prompt or (
            "You are a careful software testing assistant. "
            "Generate high-quality unit tests that are deterministic, minimal, and relevant."
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name, e.g., 'zero_shot', 'few_shot', 'cot', 'tot', 'gtot'."""
        raise NotImplementedError

    @property
    def default_temperature(self) -> float:
        """
        Sensible default for test generation.
        You can override in subclasses.
        """
        return 0.2

    @property
    def default_max_tokens(self) -> int:
        return 1200

    # ---- Public API ----

    def build(self, task: Task, **kwargs: Any) -> PromptBundle:
        """
        Build a prompt bundle for a single generation call.
        For ToT/GToT, you can still use this as the first call (root step),
        and then run iterative steps using step()/run().
        """
        messages = self._build_messages(task, **kwargs)
        bundle = PromptBundle(
            framework=self.name,
            task_id=task.task_id,
            messages=messages,
            extra={"kwargs": kwargs},
            trace=[],
        )
        return bundle

    @abstractmethod
    def _build_messages(self, task: Task, **kwargs: Any) -> List[Message]:
        """Framework-specific message construction."""
        raise NotImplementedError

    def _base_vars(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """
        Default template variables available to all frameworks.
        Subclasses can extend.
        """
        return {
            "task_id": task.task_id,
            "language": task.language,
            "source_code": task.source_code,
            "prompt_text": task.prompt_text,
            **kwargs,
        }

    def _system_message(self) -> Message:
        return Message(role="system", content=self.system_prompt)