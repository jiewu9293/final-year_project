"""
Dataset loader for JSON array or JSONL format files.
Supports filtering and explicitly removes leakage fields like 'tests'.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


LEAKAGE_FIELDS: Set[str] = {'tests', 'test_list'}


def load_json_array_or_jsonl(
    path: str,
    limit: Optional[int] = None,
    offset: int = 0,
    task_ids: Optional[List[str]] = None,
    remove_fields: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON array or JSONL file.
    
    Args:
        path: Path to the dataset file (.jsonl or .json)
        limit: Maximum number of items to load (None = all)
        offset: Number of items to skip from the beginning
        task_ids: If provided, only load items with these task_ids
        remove_fields: Additional fields to remove (default: LEAKAGE_FIELDS)
    
    Returns:
        List of dataset items with leakage fields removed
    
    Example:
        >>> data = load_json_array_or_jsonl("ULT.jsonl", limit=10)
        >>> assert 'tests' not in data[0]
        >>> assert 'code' in data[0]
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    with open(path_obj, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    if task_ids is not None:
        task_id_set = set(str(tid) for tid in task_ids)
        data = [item for item in data if str(item.get('task_id', '')) in task_id_set]
    
    data = data[offset:]
    
    if limit is not None:
        data = data[:limit]
    
    fields_to_remove = remove_fields or LEAKAGE_FIELDS
    
    cleaned_data = []
    for item in data:
        cleaned_item = {k: v for k, v in item.items() if k not in fields_to_remove}
        cleaned_data.append(cleaned_item)
    
    return cleaned_data
