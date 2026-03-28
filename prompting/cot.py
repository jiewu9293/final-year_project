"""
Chain-of-Thought (CoT) prompting framework for test generation.

Single-step approach that instructs the LLM to reason step-by-step
before generating test code. Unlike ToT/GToT, this uses only one LLM call
but includes explicit reasoning instructions in the prompt.
"""

from .base import PromptFramework, Message


class CoTPrompting(PromptFramework):
    """
    Chain-of-Thought prompting: reason step-by-step, then generate tests.

    Architecture:
    - Single LLM call with structured reasoning instructions
    - LLM first analyzes the code, then generates tests
    """

    @property
    def name(self) -> str:
        return "cot"

    def _build_messages(self, task, **kwargs):
        vars = self._base_vars(task, **kwargs)
        user_prompt = self.renderer.render("cot.txt", **vars)

        return [
            self._system_message(),
            Message(role="user", content=user_prompt),
        ]