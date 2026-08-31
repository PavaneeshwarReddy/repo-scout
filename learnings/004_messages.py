"""
Messages: 
- role : who actually posted that message
- content: actual data 
- metadata: something like tokens consumed, messageIds and other


Message types:
- ToolMessage : When model makes tool calls they are included in AIMessage
- AIMessage : output of a model invocation
- HumanMessage : user input and interaction
- SystemMessage : intial set of instructions given to the model, defienes the model's role

Content: Entire group of content blocks
Content Blocks: Response can contain multi-modal data which is useful for differentiating between responses

"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model

# level - 0
# types of messages

def message1():
    messages = [
        SystemMessage("You are a traveller"),
        HumanMessage("Favourite place in south goa ? "),
        AIMessage("Butterfly beach"),
        HumanMessage("Describe it"),

    ]

    model = init_chat_model("ollama:llama3.2")
    response = model.invoke(messages)
    print(response.content)
    print(response.usage_metadata) # contains information about tokens used and generated


# level - 1
# defining tool calls manually
def message2():

    model = init_chat_model("ollama:llama3.2")

    ai_message = AIMessage(
        content=[],
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "knr"},
                "id": "call123"
            }
        ]
    )

    tool_message = ToolMessage(
        content="23 degrees celcius",
        tool_call_id="call123" # should match with the ai message call id
    )

    messages = [
        HumanMessage("whats the weather in knr ?"),
        ai_message,
        tool_message
    ]

    response = model.invoke(messages)
    print(response.content)


# level-3
# content blocks 
def message3():
    # provider specific, this content is later parsed and converted to content-blocks lazily
    human_message = HumanMessage(content=[
        {"type": "text", "text": "Hello, how are you?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ])

    # List of standard content blocks
    human_message = HumanMessage(content_blocks=[
        {"type": "text", "text": "Hello, how are you?"},
        {"type": "image", "url": "https://example.com/image.jpg"},
    ])


# level - 4
# serialization
from langchain_core.load import dumpd, load

def message4():
    human_msg = HumanMessage("can you describe goa ?")
    raw_msg = dumpd(human_msg)
    human_msg = load(raw_msg)
