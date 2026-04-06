# Day 4 – Multi-Step Planning Agent

## Objective
Design an agent that decomposes a task into sequential steps, executes them one by one, and shows intermediate outputs at each stage.

## File Structure
```
day4/
├── planner.py  # Step definitions + pipeline runner
└── agent.py    # Entry point / REPL
```

## Pipeline
```
User Query
    │
    ▼
Step 1 – Extract Numbers    → [5.0, 10.0, 15.0]
    │
    ▼
Step 2 – Compute Average    → 10.00
    │
    ▼
Step 3 – Generate Summary   → "The average of 3 number(s) (5, 10, 15) is 10.00."
```

## How to Run
```bash
python agent.py
```

## Example Session
```
You: Find the average of 5, 10, 15 and summarize the result

────────────────────────────────────────────────────
Query: Find the average of 5, 10, 15 and summarize the result
────────────────────────────────────────────────────

Step 1 – Extract Numbers
  → Extracted numbers: [5.0, 10.0, 15.0]

Step 2 – Compute Average
  → Average of [5.0, 10.0, 15.0] = 10.00

Step 3 – Generate Summary
  → The average of the 3 number(s) (5, 10, 15) is 10.00.

────────────────────────────────────────────────────

Agent (final answer): The average of the 3 number(s) (5, 10, 15) is 10.00.
```
