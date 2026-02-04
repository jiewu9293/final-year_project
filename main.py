from prompting.zero_shot import ZeroShotPrompting
from prompting.base import Task
from clients.openai_client import OpenAIClient

import os
import time

from utils.io import append_jsonl

def main():

    task = Task(
        task_id="debug_is_even",
        language="python",
        source_code="""
def is_even(n: int) -> bool:
    return n % 2 == 0
""".strip(),
        prompt_text="Write pytest unit tests for the function above.",
        metadata={}
    )


    framework = ZeroShotPrompting()


    client = OpenAIClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2
    )


    result = framework.run(
        client=client,
        task=task,
    )

    
    print("===== GENERATED TESTS =====")
    print(result["output"])

    output_path = os.path.join("outputs", "zero_shot_tests.py")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["output"])
    print(f"Saved to: {output_path}")

    record = {
        "timestamp": time.time(),
        "task_id": task.task_id,
        "model": getattr(client, "last_model", None) or client.model,
        "prompting": framework.name,
        "input_tokens": getattr(client, "last_input_tokens", None),
        "output_tokens": getattr(client, "last_output_tokens", None),
        "total_tokens": getattr(client, "last_total_tokens", None),
        "latency": getattr(client, "last_latency", None),
        "output_file": output_path,
    }
    append_jsonl(os.path.join("results", "results.jsonl"), record)

if __name__ == "__main__":
    main()
