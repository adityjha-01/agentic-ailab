"""
Assignment 1: Rule-Based AI Agent
A simple agent that identifies user intent via keyword matching and performs actions.
"""

import datetime
import re


# ── Input Handler ─────────────────────────────────────────────────────────────

def get_input(prompt: str = "You: ") -> str:
    return input(prompt).strip()


# ── Intent Detection ──────────────────────────────────────────────────────────

def detect_intent(user_input: str) -> str:
    text = user_input.lower()

    if any(k in text for k in ["hello", "hi", "hey", "greet"]):
        return "greet"
    if any(k in text for k in ["calculate", "compute"]) or re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
        return "calculate"
    if any(k in text for k in ["date", "today", "day", "time"]):
        return "datetime"
    if any(k in text for k in ["bye", "exit", "quit", "goodbye"]):
        return "exit"
    return "unknown"


# ── Action Execution ──────────────────────────────────────────────────────────

def execute_action(intent: str, user_input: str) -> str:
    if intent == "greet":
        return "Hello! How can I assist you today?"

    if intent == "calculate":
        expr = re.search(r"[\d\s\+\-\*\/\.]+", user_input)
        if expr:
            try:
                result = eval(expr.group().strip())
                return f"Result: {result}"
            except Exception:
                pass
        return "Sorry, I couldn't parse that calculation."

    if intent == "datetime":
        now = datetime.datetime.now()
        return f"Current date and time: {now.strftime('%A, %d %B %Y  %H:%M:%S')}"

    if intent == "exit":
        return "Goodbye!"

    return "I'm not sure how to help. Try: 'hello', 'calculate 5+3', or 'date'."


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_agent():
    print("=== Rule-Based AI Agent (Day 1) ===")
    print("Type 'exit' or 'bye' to quit.\n")

    while True:
        user_input = get_input()
        if not user_input:
            continue

        intent = detect_intent(user_input)
        response = execute_action(intent, user_input)
        print(f"Agent: {response}\n")

        if intent == "exit":
            break


if __name__ == "__main__":
    run_agent()
