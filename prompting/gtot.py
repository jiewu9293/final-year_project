"""
Graph-of-Thoughts (GToT) prompting framework for test generation.

Multi-step approach:
1. Analyze code structure and identify nodes (branches, conditions, paths)
2. Generate test strategies for each node and their combinations
3. Synthesize final test code based on strategies

Each step is a separate LLM call, with intermediate results stored in trace.
"""

from typing import Any, Dict, List
from .base import PromptFramework, Message, Task, PromptBundle, LLMClient


class GToTPrompting(PromptFramework):
    """
    Graph-of-Thoughts prompting with explicit multi-step reasoning.
    
    Architecture:
    - Step 1: Code analysis → identify nodes and edges
    - Step 2: Strategy generation → test plan for each node/combination
    - Step 3: Code synthesis → final pytest code
    """

    @property
    def name(self) -> str:
        return "gtot"

    @property
    def default_temperature(self) -> float:
        # Slightly higher for creative graph exploration
        return 0.3

    def _build_messages(self, task: Task, **kwargs) -> List[Message]:
        """
        Build initial messages for Step 1 (code analysis).
        For multi-step execution, use run_multistep() instead.
        """
        vars = self._base_vars(task, **kwargs)
        user_prompt = self.renderer.render("gtot_step1.txt", **vars)
        
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
        Execute the full 3-step GToT pipeline.
        
        Args:
            task: The task to generate tests for
            client: LLM client for making API calls
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            PromptBundle with final messages and trace of all steps
        """
        trace = []
        vars = self._base_vars(task, **kwargs)
        
        # Step 1: Analyze code structure
        step1_prompt = self.renderer.render("gtot_step1.txt", **vars)
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
            "name": "code_analysis",
            "prompt": step1_prompt,
            "response": step1_response,
        })
        
        # Step 2: Generate test strategies
        vars["analysis"] = step1_response
        step2_prompt = self.renderer.render("gtot_step2.txt", **vars)
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
            "name": "strategy_generation",
            "prompt": step2_prompt,
            "response": step2_response,
        })
        
        # Step 3: Synthesize test code
        vars["strategies"] = step2_response
        step3_prompt = self.renderer.render("gtot_step3.txt", **vars)
        step3_messages = [
            self._system_message(),
            Message(role="user", content=step3_prompt),
        ]
        
        step3_response = client.generate(
            [{"role": m.role, "content": m.content} for m in step3_messages],
            **kwargs
        )
        
        trace.append({
            "step": 3,
            "name": "code_synthesis",
            "prompt": step3_prompt,
            "response": step3_response,
        })
        
        # Final bundle contains Step 3 messages (for compatibility)
        # but trace contains all intermediate steps
        bundle = PromptBundle(
            framework=self.name,
            task_id=task.task_id,
            messages=step3_messages + [Message(role="assistant", content=step3_response)],
            extra={
                "kwargs": kwargs,
                "num_steps": 3,
                "final_response": step3_response,
            },
            trace=trace,
        )
        
        return bundle
