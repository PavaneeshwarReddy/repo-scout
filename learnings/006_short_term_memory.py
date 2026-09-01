"""
Short term Memory: 
This is the state which we use, after the conversation ended it is removed, within the same thread or conversation
Useful whenever a server restart happens and you want to start with the context again

While we are trying to only retain short term meaning less messages we can do all these
- Trim Messages
- Delete Messages
- Summarize Messages
- Custom Strategies

before_model : human -> before_model -> model -> tools -> before_model -> model
after_model : human -> model -> after_model -> tools -> model

"""

# level - 0 
# short term memory
from langgraph.checkpoint.memory import InMemorySaver # postgres has to be used in production only for testing
from langchain.agents import create_agent

def stm1():
    checkpointer = InMemorySaver()

    agent = create_agent(
       "ollama:llama3.2",
       checkpointer=checkpointer
    )

    agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "can you tell what is love ?"
            }
        ]
    }, config={"configurable": {"thread_id": "1234"}})

    print(checkpointer)

stm1()

# level - 1
# trim messages
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime


def stm2():

    @before_model # just called before invoking the model
    def trim_messages(state: AgentState, runtime: Runtime):
        messages = state["messages"]

        new_messages = messages[3:] # remove the older messages and add recent messages

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }


    agent = create_agent(
       "ollama:llama3.2",
        middleware=[trim_messages]
    )


# level - 2
# delete messages : don't remember any conversation
from langchain.agents.middleware import after_model

def stm3():
    @after_model
    def delete_messages(state: AgentState, runtime: Runtime):
        messages = state["messages"]
        return {
            "messages": [
                RemoveMessage(id=[m.id for m in messages])
            ]
        }


    agent = create_agent(
       "ollama:llama3.2",
        middleware=[delete_messages]
    )


# level - 3
# summarization of previous messages
from langchain.agents.middleware import SummarizationMiddleware

def stm4():

    checkpointer = InMemorySaver()

    agent = create_agent(
       "ollama:llama3.2",
        middleware=[
            SummarizationMiddleware(
                "ollama:llama3.2",
                keep=("messages", 20), # summarize and keep last 20 messages
                trigger=("tokens", 4000), # whenever input tokens increase more than 4000 this summarizes

            )
        ],
        checkpointer=checkpointer
    )


# level - 4
# dynamic prompts

from langchain.agents.middleware import dynamic_prompt, ModelRequest

def stm5():

    @dynamic_prompt
    def dynamic_prompts(request: ModelRequest):
        user_name = request.state.get("user_name")
        return f"You are very helpful assitant, address user as {user_name}"

    agent = create_agent(
       "ollama:llama3.2",
        middleware=[dynamic_prompt]
    )

