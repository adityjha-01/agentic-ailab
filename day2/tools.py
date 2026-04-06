"""
Assignment 2 – Tool Definitions
Each tool is a self-contained callable that accepts a string and returns a string.
"""

import re


def calculator_tool(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    expr = re.search(r"[\d\s\+\-\*\/\.\(\)]+", expression)
    if expr:
        try:
            result = eval(expr.group().strip())
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    return "No valid expression found."


def weather_tool(city: str) -> str:
    """Return mocked weather data for a city."""
    mock_data = {
        "mumbai":   "🌤 Mumbai: 32°C, Humid, partly cloudy",
        "delhi":    "🌬 Delhi: 28°C, Windy, clear skies",
        "london":   "🌧 London: 14°C, Rainy, overcast",
        "new york": "🌥 New York: 18°C, Cool, light breeze",
        "tokyo":    "🌸 Tokyo: 22°C, Mild, cherry blossoms",
    }
    key = city.lower().strip()
    return mock_data.get(key, f"No weather data available for '{city}'.")


def summarizer_tool(text: str) -> str:
    """Produce a short summary: first sentence + word count."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    first = sentences[0] if sentences else text
    words = len(text.split())
    return f"Summary: {first} [Total words: {words}]"


# Registry: intent → (tool_function, argument_extractor)
TOOLS = {
    "calculate": (calculator_tool, lambda q: q),
    "weather":   (weather_tool,    lambda q: re.sub(r"(?i)weather\s*(in|for|at)?\s*", "", q).strip()),
    "summarize": (summarizer_tool, lambda q: re.sub(r"(?i)^(summarize|summary of|summarise)\s*", "", q).strip()),
}
