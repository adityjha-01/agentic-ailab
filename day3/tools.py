"""
Day 3 – shared tool definitions (reused for LLM agent).
"""

import re


def calculator_tool(expression: str) -> str:
    expr = re.search(r"[\d\s\+\-\*\/\.\(\)]+", expression)
    if expr:
        try:
            return f"Calculation result: {eval(expr.group().strip())}"
        except Exception as e:
            return f"Calculation error: {e}"
    return "No valid expression found."


def weather_tool(city: str) -> str:
    mock = {
        "mumbai":   "🌤 Mumbai: 32°C, Humid",
        "delhi":    "🌬 Delhi: 28°C, Windy",
        "london":   "🌧 London: 14°C, Rainy",
        "new york": "🌥 New York: 18°C, Cool",
        "tokyo":    "🌸 Tokyo: 22°C, Mild",
    }
    return mock.get(city.lower().strip(), f"No data for '{city}'.")


def summarizer_tool(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    first = sentences[0] if sentences else text
    return f"Summary: {first} [Words: {len(text.split())}]"


TOOLS = {
    "calculator": calculator_tool,
    "weather":    weather_tool,
    "summarizer": summarizer_tool,
}
