from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_core.utils.uuid import uuid7
from pydantic import BaseModel
from dataclasses import dataclass

"""
Agent:
Which calls tools in a loop until task is completed


Models: 
- A configurable model instance which can be passed to agent to perform task

Tools:
- Basic function calls that can be executed by the model to get some context

System Prompts:
- Behaviour and boundaries that can be configured

Agent State:
- Every agene maitains it' own execution state, which stores current convo history and some fields for tools and middle ware
- results["messages"] - Maintains every convo history, never replaces it, just appends
- Same thread_id have the same conversation history and can be passed in config

Context:
- If you want to pass API_KEYS or USER_IDS you can use this
- You can pass in create agent itself


Streaming:
- If we want the model current all processes we can use this


Middlewares:
- For more control, we can use these
- They can be useful in providing additional tools to the agent
- Agent needs different features such as 
    - Execution Environment
    - Context Management
    - Planning and Delegation
    - Fault Tolerence
    - Guardrails
    - Steering
"""

class ResponseFormat(BaseModel):
    name: str
    stocks: list

@dataclass
class ContextSchema:
    GOOGLE_MAPS_API_KEY: str


config = {
    "configurable": {"thread_id": str(uuid7)}
}

agent1 = create_agent(
    model="ollama:llama3.2", # provider:model
    tools=[], # any callable python function or langchain tool or tool dict
    system_prompt="", # shapes how the agent approaches to performing the tasks
    response_format=ResponseFormat, # how you want the response to be   
    context_schema=ContextSchema, # you can pass the context schema
    name="pavan-1st-agent" # name given to the agent
)

# while invoking you can pass config and context
agent1.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "xyz"
            }
        ]
    }, config=config, context=ContextSchema(GOOGLE_MAPS_API_KEY="12342asdasd")
)

# this contains same thread_id so it appends to the same message history
agent2 = create_agent(
    model="ollama:llama3.2", # provider:model
    tools=[], # any callable python function or langchain tool or tool dict
    system_prompt="", # shapes how the agent approaches to performing the tasks
    response_format=ResponseFormat, # how you want the response to be   
)


# while invoking you can pass config and context, same config has same chat history
agent2.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "xyz"
            }
        ]
    }, config=config, context=ContextSchema(GOOGLE_MAPS_API_KEY="12342asdasd")
)


# stream events
stream = agent1.stream_events(
    {
        "messages": [
            {
                "role": "user",
                "content": "Nothing much important"
            }
        ]
    },
    version="v3"
)

# to stream pick the latest event and see the status
for snapshot in stream.values:
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


# --------------- Middlewares ------------------
"""
Execution Environment
- Provides agent a workspace, file system ( r and w ) , tools, executing shell commands and more
"""
from deepagents import FilesystemMiddleware # provides other tools like read write file and more
from deepagents.backends import StateBackend # stores files agent ephermal, not shared 

agent3 = create_agent(
    model="ollama:llama3.2",
    tools=[],
    middleware=[FilesystemMiddleware(backend=StateBackend())]
)


"""
Context Management
- Every agent creates new messages context which can overflow
- This helps in even summarizing context when passing through next
- This can involce summarization, skills, memory middlware
"""
from deepagents.middleware import SummarizationMiddleware, SkillsMiddleware, MemoryMiddleware
backend = StateBackend()
model="ollama:llama3.2"

agent4 = create_agent(
    model=model,
    tools=[],
    middleware=[ 
        FilesystemMiddleware(backend=backend), # gives access to file systems
        SummarizationMiddleware(model=model), # which automatically summarizes for more context window hits
        SkillsMiddleware(backend=backend, sources=["./skills/"]), # different skill access
        MemoryMiddleware(backend=backend, sources=['./AGENT.md']) # this can load this file on startup and pass to it as context
    ]
)

"""
Planning and Deligation
- Delegation helps in breaking main agent work into peices and handing it over to subagents ( their own isolated context )
- Work can run in parallel but main agent context stays clean
- A subagent is selected by main agent by it's tool capabilities
"""
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
agent5 = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[
            {
                "name": "reasearch-agent",
                "description": "can research better",
                "tools": [],
                "model": model,
                "middleware": []
            }
        ])
    ]
)


"""
Fault tolerence
-  Model Failures and Tool failures and their max retries
- When agent calls a model if is unavialable or it a model calls tool and it returns error
"""
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

agent6 = create_agent(
    model=model, 
    tools=[],
    middleware=[
        ModelRetryMiddleware(max_retries=2),
        ToolRetryMiddleware(max_retries=2)
    ]
)


"""
Guadrails
- All policies can't be restricted in the prompt
- Some restrictions has to on the go, based on the middle ware we can either block, restrict
"""
from langchain.agents.middleware import PIIMiddleware

agent7 = create_agent(
    model=model,
    tools=[],
    middleware=[
        PIIMiddleware(pii_type="mac_address", strategy='block')
    ]
)


"""
Steering
- Some tasks need human intervention
"""
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent8 = create_agent(
    model=model,
    tools=[],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
    ]
)

