"""
Assignment 4 – Multi-Step Agent (Planning)
Accepts a user query and runs it through a sequential planning pipeline.
"""

from planner import run_pipeline


def run_agent():
    print("=== Multi-Step Planning Agent (Day 4) ===")
    print("Example: 'Find the average of 5, 10, 15 and summarize the result'")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Agent: Goodbye!")
            break

        final = run_pipeline(user_input)
        print(f"Agent (final answer): {final}\n")


if __name__ == "__main__":
    run_agent()
