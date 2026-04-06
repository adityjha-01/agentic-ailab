"""
Assignment 2 – Tool-Using Agent
Detects which tool to use, extracts the argument, calls it, and returns the result.
"""

import re
from tools import TOOLS


def detect_tool(user_input: str) -> str:
    text = user_input.lower()
    if any(k in text for k in ["calculate", "compute"]) or re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
        return "calculate"
    if "weather" in text:
        return "weather"
    if any(k in text for k in ["summarize", "summarise", "summary"]):
        return "summarize"
    return "unknown"


def run_agent():
    print("=== Tool-Using Agent (Day 2) ===")
    print("Tools: calculator | weather <city> | summarize <text>")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Agent: Goodbye!")
            break

        tool_name = detect_tool(user_input)
        if tool_name == "unknown":
            print("Agent: I don't know which tool to use. Try 'calculate', 'weather', or 'summarize'.\n")
            continue

        tool_fn, arg_extractor = TOOLS[tool_name]
        argument = arg_extractor(user_input)
        result = tool_fn(argument)
        print(f"Agent [{tool_name}]: {result}\n")


if __name__ == "__main__":
    run_agent()
