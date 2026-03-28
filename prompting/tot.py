"""
Tree-of-Thoughts (ToT) prompting framework for test generation.

Multi-step approach:
1. Generate multiple candidate test strategies (branching)
2. Evaluate and select the best strategy, then synthesize final test code

Each step is a separate LLM call, with intermediate results stored in trace.
"""

from typing import Any, Dict, List
from .base import PromptFramework, Message, Task, PromptBundle, LLMClient


class ToTPrompting(PromptFramework):
    """
    Tree-of-Thoughts prompting with branching and evaluation.

    Architecture:
    - Step 1: Generate multiple candidate test strategies (branching)
    - Step 2: Evaluate strategies and synthesize final pytest code
    """

    @property
    def name(self) -> str:
        return "tot"

    @property
    def default_temperature(self) -> float:
        return 0.3

    def _build_messages(self, task: Task, **kwargs) -> List[Message]:
        """
        Build initial messages for Step 1 (strategy generation).
        For multi-step execution, use run_multistep() instead.
        """
        vars = self._base_vars(task, **kwargs)
        user_prompt = self.renderer.render("tot_step1.txt", **vars)

        return [
            self._system_message(),
            Message(role="user", content=user_prompt),
        ]

    def run_multistep(
        self,
        task: Task,
        client: LLMClient,
        **kwargs: Any
    ) -> PromptBundle:
        """
        Execute the full 2-step ToT pipeline.

        Args:
            task: The task to generate tests for
            client: LLM client for making API calls
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            PromptBundle with final messages and trace of all steps
        """
        trace = []
        vars = self._base_vars(task, **kwargs)

        # Step 1: Generate multiple candidate test strategies
        step1_prompt = self.renderer.render("tot_step1.txt", **vars)
        step1_messages = [
            self._system_message(),
            Message(role="user", content=step1_prompt),
        ]

        step1_response = client.generate(
            [{"role": m.role, "content": m.content} for m in step1_messages],
            **kwargs
        )

        trace.append({
            "step": 1,
            "name": "strategy_generation",
            "prompt": step1_prompt,
            "response": step1_response,
        })

        # Step 2: Evaluate strategies and synthesize test code
        vars["strategies"] = step1_response
        step2_prompt = self.renderer.render("tot_step2.txt", **vars)
        step2_messages = [
            self._system_message(),
            Message(role="user", content=step2_prompt),
        ]

        step2_response = client.generate(
            [{"role": m.role, "content": m.content} for m in step2_messages],
            **kwargs
        )

        trace.append({
            "step": 2,
            "name": "evaluate_and_synthesize",
            "prompt": step2_prompt,
            "response": step2_response,
        })

        bundle = PromptBundle(
            framework=self.name,
            task_id=task.task_id,
            messages=step2_messages + [Message(role="assistant", content=step2_response)],
            extra={
                "kwargs": kwargs,
                "num_steps": 2,
                "final_response": step2_response,
            },
            trace=trace,
        )

        return bundle