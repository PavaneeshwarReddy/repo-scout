from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
import urllib.request
import urllib.error

"""
Agent = Model + Harness ( create_agent: prompt, tools, middleware )

Tools: 
- Functions that can be invoked by the agent, params are decided by agent
- If there are many tools, agent decides based on func name, doc string, params and return types

System Prompts:
- Defines rules and behaviour
- It can define which tool incase of which use case, and a fallback if it is unable to execute that tool

Results ( Mostly the same format ):
- res = results["messages"] - contain most of the information we needed
- res[0] - HumanMessage -> question asked, any kwargs sent
- res[1] - AIMessage ( execution of tool call ) ->  tool_calls with what args did it make a tool call
         - This contains output tokens and input tokens
- res[2] - ToolMessage -> what did tool call return
- res[3] (Actual result which we are interested ) - AIMessage -> Final message that is constructed by using tool call output and system prompt
         - This contains output tokens and input tokens

Multiple tool calls can increase the indices but last always contains the required result

Messages & Roles:
system    : instructions for the AI
user      : request from the user
assistant : previous response from the AI
tool      : result returned by a tool

Memory: 
Instate memory that remembers and passed to other agents


Model:
Intiate a model with custom params to tune the model for our use case
temperature - uses more possible tokens and explores creative writing ( 0-1 )

Deepagents:
- It internally creates sub agents to perform tasks, write-todods, plan and execute
- It's similar to normal agents with some guided plans 
"""


# ---------- QUICK START ( High Level Intros to All Features ) -----------

# level - 0
# Invoke a agent with a static tool call
def agent_1():
    def get_weather(city:str) -> str:
        """Get weather based on city name"""
        return "23 degrees celcius"
    
    agent = create_agent(
        model="ollama:llama3.2",
        tools=[get_weather],
        system_prompt="""
            You are a weather assistant, 
            Whenever users asks information related to weather invoke tool call get_weather and get the information
            If the tool fails then return unable to fetch whether
        """
    )

    results = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What the whether situation in karimnagar ?"
            }
        ]
    })
    print(results["messages"][-1].content_blocks)


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

# level - 1
# Invoke a agent with dynamic tool call
def agent_2():
    SYSTEM_PROMPT = """ Consider you are a real quant researcher
    - fetch_text_from_url -> this takes in url from where we want to pull in the extract text
    """

    agent = create_agent(
        model="ollama:llama3.2",
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT
    )

    results = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Can you get the latest trading details of the market https://www.nseindia.com/ "
            }
        ]
    })

    print(results["messages"][-1].content_blocks)


# level - 2 
# Instate memory persistance and passing to a deep agents
def agent_3():

    init_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.5,
        timeout=3000,
        max_tokens=25000,
    )

    checkpoint = InMemorySaver()

    agent = create_agent(
        model=init_model,
        tools=[fetch_text_from_url],
        system_prompt=""" Consider you are a real quant researcher
        - fetch_text_from_url -> this takes in url from where we want to pull in the extract text
        """,
        checkpointer=checkpoint
    )

    deep_agent = create_deep_agent(
        model=init_model,
        tools=[fetch_text_from_url],
        system_prompt=""" Consider you are a real quant researcher
        - fetch_text_from_url -> this takes in url from where we want to pull in the extract text
        """,
        checkpointer=checkpoint

    )

    content = """
        Consider this website to fetch the data  https://www.nseindia.com/
        From this url fetch market information make analyzes and give me list of things i can invest on
        Also give me what are the latest news and where there is a chance of spike in marker prices
        Also give me list of top loosers and gainers
    """

    results = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }, config={
        "configurable": {"thread_id": "normal-agent-1"}
    })

    print(results["messages"][-1].content_blocks)

    results = deep_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
      }, config={
        "configurable": {"thread_id": "deep-agent-1"}
    }
    )

    print(results["messages"][-1].content_blocks)


# level - 3
# Stream the events
def agent_4():
    SYSTEM_PROMPT = """ Consider you are a real quant researcher
    - fetch_text_from_url -> this takes in url from where we want to pull in the extract text
    """

    agent = create_agent(
        model="ollama:llama3.2",
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT
    )

    stream = agent.stream_events({
        "messages": [
            {
                "role": "user",
                "content": "Can you get the latest trading details of the market https://www.nseindia.com/ "
            }
        ]
    }, version="v3")

    for snapshot in stream.values:
        latest_message = snapshot["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"User: {latest_message.content}")
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


