"""
Streams
Less consise then event streaming, not much of a use we mostly use stream events

Common patterns
- Stream agent progress
- stream llm tokens
- stream thinking
- stream multiple modes
"""



from langchain.tools import tool
from langchain.agents import create_agent


def s1():
    @tool
    def get_weather(city: str) -> str:
        """Returns weather based on the city provided"""
        return "23 degrees celcius"

    @tool
    def get_location()-> str:
        """When user asks for his location, this function is used and returns the city"""
        return "Karimnagar"

    agent = create_agent(
        "ollama:llama3.2",
    )

    messages = [
        {
            "role": "user",
            "content": "which city iam in ?"
        },
        {
            "role": "user",
            "content": "what is the weather outside ? "
        }
    ]

    streams = agent.stream({
        "messages": messages
    },  stream_mode="messages")

    result = ""

    for message, metadata in streams:
        print(message.content)

s1()
