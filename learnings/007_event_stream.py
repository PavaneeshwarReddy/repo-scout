"""
Event Streaming
- It gives individual events what are happening in the system
- something like tool start, tool end , model invoked ,



stream.messages	Model message streams, one per LLM call.
message.text	Text deltas and final text for a message.
message.reasoning	Reasoning deltas for models that expose reasoning content.
message.tool_calls	Tool-call argument chunks and finalized tool calls.
message.output	Final message object after the model call completes.
stream.values	Agent state snapshots.
stream.output	Final agent state.
stream.subgraphs	Nested graph runs (sub-agents and plain subgraphs).
stream.extensions	Custom transformer projections.
stream.tool_calls	Tool execution lifecycle, inputs, output deltas, final output, and errors.
"""

from langchain.tools import tool
from langchain.agents import create_agent

# level - 0
# stream events
def es1():
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

    for stream in agent.stream_events({"messages": messages}, version="v3"):
        print(stream)

