"""
Assignment 4 – Multi-Step Planner
Breaks a complex query into discrete steps, executes them sequentially,
and prints intermediate outputs at each stage.
"""

import re
import statistics


# ── Individual Steps ──────────────────────────────────────────────────────────

def step_extract_numbers(query: str) -> dict:
    """Step 1: Pull all numbers from the user query."""
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", query)]
    return {
        "numbers": numbers,
        "output":  f"Extracted numbers: {numbers}",
    }


def step_compute_average(data: dict) -> dict:
    """Step 2: Compute the arithmetic mean of the extracted numbers."""
    nums = data.get("numbers", [])
    if not nums:
        return {**data, "average": None, "output": "No numbers to average."}
    avg = statistics.mean(nums)
    return {
        **data,
        "average": avg,
        "output":  f"Average of {nums} = {avg:.2f}",
    }


def step_generate_summary(data: dict) -> dict:
    """Step 3: Produce a human-readable summary of the result."""
    avg  = data.get("average")
    nums = data.get("numbers", [])

    if avg is None:
        summary = "Could not compute an average (no numbers found)."
    else:
        nums_str = ", ".join(str(int(n) if n == int(n) else n) for n in nums)
        summary  = (
            f"The average of the {len(nums)} number(s) "
            f"({nums_str}) is {avg:.2f}."
        )

    return {**data, "summary": summary, "output": summary}


# ── Pipeline Definition ───────────────────────────────────────────────────────

PIPELINE = [
    ("Extract Numbers",  step_extract_numbers),
    ("Compute Average",  step_compute_average),
    ("Generate Summary", step_generate_summary),
]


def run_pipeline(query: str) -> str:
    """Execute the pipeline and print each intermediate result."""
    separator = "─" * 52
    print(f"\n{separator}")
    print(f"Query: {query}")
    print(separator)

    state = query   # first step receives the raw string
    for i, (name, fn) in enumerate(PIPELINE, start=1):
        result = fn(state)
        state  = result
        print(f"\nStep {i} – {name}")
        print(f"  → {result['output']}")

    print(f"\n{separator}\n")
    return state.get("summary", "Pipeline complete.")
