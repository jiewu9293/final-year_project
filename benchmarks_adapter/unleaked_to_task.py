"""
Adapter to convert UnLeakedTestBench dataset items to Task objects.
"""

from prompting.base import Task
from typing import Dict, Any


def item_to_task(item: Dict[str, Any], benchmark_name: str = "ult") -> Task:
    """
    Convert a ULT/PLT dataset item to a Task object.
    
    Args:
        item: Dataset item with fields like:
            - task_id: unique identifier
            - code: function/class code under test
            - prompt: task description
            - func_name: function name (optional)
            - cyclomatic_complexity: complexity metric (optional)
        benchmark_name: Prefix for task_id (default: "ult")
    
    Returns:
        Task object ready for prompt framework
    
    Example:
        >>> item = {
        ...     "task_id": "1",
        ...     "code": "def foo(): pass",
        ...     "prompt": "Test this function",
        ...     "func_name": "foo"
        ... }
        >>> task = item_to_task(item, benchmark_name="ult")
        >>> assert task.task_id == "ult_1"
        >>> assert task.source_code == "def foo(): pass"
    """
    task_id = f"{benchmark_name}_{item['task_id']}"
    
    source_code = item['code']
    prompt_text = item['prompt']
    
    metadata = {
        'func_name': item.get('func_name', ''),
        'cyclomatic_complexity': item.get('cyclomatic_complexity'),
        'original_task_id': item['task_id'],
        'benchmark': benchmark_name
    }
    
    task = Task(
        task_id=task_id,
        language="python",
        source_code=source_code,
        prompt_text=prompt_text,
        metadata=metadata
    )
    
    return task
