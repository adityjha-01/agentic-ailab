# Day 3 – LLM-Based Agent

## Objective
Replace hand-coded intent detection with an LLM that reads the query and decides which tool to invoke.

## File Structure
```
day3/
├── tools.py      # Reusable tool functions
├── agent.py      # LLM router + main loop
└── agent_log.txt # Auto-generated run log
```

## Setup (optional – works without API key too)
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```
If no key is set, the agent falls back to a keyword-based simulated router automatically.

## How to Run
```bash
python agent.py
```

## Logging
Every turn is appended to `agent_log.txt`:
```json
{"timestamp": "2026-04-06T10:30:00", "input": "...", "tool": "calculator", "argument": "15*4", "output": "..."}
```

## Example Session
```
You: What's 99 divided by 3?
Agent [calculator]: Calculation result: 33.0

You: weather in London
Agent [weather]: 🌧 London: 14°C, Rainy

You: summarize Machine learning is a subset of artificial intelligence.
Agent [summarizer]: Summary: Machine learning is a subset of artificial intelligence. [Words: 9]
```
