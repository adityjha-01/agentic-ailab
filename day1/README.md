# Day 1 – Rule-Based AI Agent

## Objective
Build a simple agent that maps user text to an intent and executes a corresponding action.

## Architecture
```
User Input → detect_intent() → execute_action() → Response
```

| Module | Responsibility |
|--------|---------------|
| `get_input()` | Reads and strips user text |
| `detect_intent()` | Keyword/regex matching → intent label |
| `execute_action()` | Dispatches to the right handler |

## Supported Intents
| Trigger words | Intent | Action |
|---|---|---|
| hello / hi / hey | `greet` | Friendly greeting |
| calculate / compute / arithmetic | `calculate` | Evaluates arithmetic |
| date / today / time | `datetime` | Prints current date-time |
| bye / exit / quit | `exit` | Ends the session |

## How to Run
```bash
python agent.py
```

## Example Session
```
You: hello
Agent: Hello! How can I assist you today?

You: calculate 12 * 7
Agent: Result: 84

You: date
Agent: Current date and time: Monday, 06 April 2026  10:30:00
```
