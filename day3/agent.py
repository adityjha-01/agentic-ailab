"""
Assignment 3 – LLM-Based Agent
Uses the Anthropic API to decide which tool to call.
Falls back to a simulated keyword router if no API key is set.
"""

import os
import re
import json
import datetime
from tools import TOOLS

# ── LLM System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a routing agent. Given a user query, decide which tool to call.
Available tools: calculator, weather, summarizer, none.
Reply ONLY with a JSON object: {"tool": "<tool_name>", "argument": "<extracted_arg>"}
- calculator: math expressions (argument = the expression)
- weather: city name lookup (argument = city name only)
- summarizer: text to summarise (argument = the raw text)
- none: greetings or unknown requests (argument = "")
No preamble. No markdown fences. JSON only."""


def llm_decide(query: str) -> dict:
    """Call Claude via Anthropic API to decide tool + argument."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except KeyError:
        print("[INFO] ANTHROPIC_API_KEY not set — using simulated LLM.")
        return _simulated_llm(query)
    except ImportError:
        print("[INFO] anthropic package not installed — using simulated LLM.")
        return _simulated_llm(query)
    except Exception as e:
        print(f"[WARN] LLM error: {e} — using simulated LLM.")
        return _simulated_llm(query)


def _simulated_llm(query: str) -> dict:
    """Keyword-based fallback when no API key is available."""
    q = query.lower()
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q) or any(w in q for w in ["calculate", "compute"]):
        return {"tool": "calculator", "argument": query}
    if "weather" in q:
        city = re.sub(r"(?i)weather\s*(in|for|at)?\s*", "", query).strip()
        return {"tool": "weather", "argument": city}
    if any(w in q for w in ["summarize", "summarise", "summary"]):
        arg = re.sub(r"(?i)^(summarize|summarise|summary of)\s*", "", query).strip()
        return {"tool": "summarizer", "argument": arg}
    return {"tool": "none", "argument": ""}


# ── Logger ────────────────────────────────────────────────────────────────────

LOG_FILE = "agent_log.txt"


def log(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_agent():
    print("=== LLM-Based Agent (Day 3) ===")
    print(f"Logs → {LOG_FILE}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Agent: Goodbye!")
            break

        decision  = llm_decide(user_input)
        tool_name = decision.get("tool", "none")
        argument  = decision.get("argument", user_input)

        if tool_name in TOOLS:
            result = TOOLS[tool_name](argument)
        else:
            result = "I'm not sure how to help with that."

        log({
            "timestamp": datetime.datetime.now().isoformat(),
            "input":     user_input,
            "tool":      tool_name,
            "argument":  argument,
            "output":    result,
        })

        print(f"Agent [{tool_name}]: {result}\n")


if __name__ == "__main__":
    run_agent()
