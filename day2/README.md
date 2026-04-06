# Day 2 – Tool-Using Agent

## Objective
Extend the rule-based agent so it delegates work to dedicated, modular tools.

## File Structure
```
day2/
├── tools.py   # Tool definitions + registry
└── agent.py   # Intent detection + tool dispatch
```

## Available Tools
| Tool | Trigger keywords | Example input |
|------|-----------------|---------------|
| `calculator_tool` | calculate / compute / arithmetic | `calculate 15 * 4` |
| `weather_tool` | weather | `weather in Tokyo` |
| `summarizer_tool` | summarize / summary | `summarize The quick brown fox...` |

## How to Run
```bash
python agent.py
```

## Example Session
```
You: calculate 100 / 4
Agent [calculate]: Calculation result: 25.0

You: weather in Mumbai
Agent [weather]: 🌤 Mumbai: 32°C, Humid, partly cloudy

You: summarize Artificial intelligence is transforming every industry worldwide.
Agent [summarize]: Summary: Artificial intelligence is transforming every industry worldwide. [Total words: 8]
```
