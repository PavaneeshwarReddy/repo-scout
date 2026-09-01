"""
Tools:
- By default functions docstring defines the model when to use it
- Args schema can be passed
- Tools can access run time information from ToolRuntime

Access Context
As tools are very powerful but when they are able to shared context like messages, file systems, etc

1. Short term Memory: only exisits until the conversation ends
2. State can be updated shared
3. Context is immutable, used to share config details, api keys, session details, etc

Tool Return types:
- String
- Object
- Command

Dynamically Selection of tools:
Sometimes tools can be very larger and this can overhelm for the model to decide

wrap_tool_call is a powerful decorater that executed before every model request, where we can use to modify the request params choose tools, etc
"""

from langchain.tools import tool

# level - 0
# basic tool decleration
def tool1():
    @tool("search_weather_tool", description="This can be used for getting live weather from supplied city")
    def get_weather(city:str)->str:
        """
            Get weather based on city
            Args:
                city: City which you want query for weather
        """
        return "23 degrees celcius"


from pydantic import BaseModel, Field

# level - 1
# define complex schema
def tool2():
    class WeatherArgs(BaseModel):
        city: str = Field(description="Which city you want to search weather for")
        day: str = Field(description="On which day you want to search for weather")
    @tool("search_weather_tool", description="This can be used for getting live weather from supplied city", args_schema=WeatherArgs)
    def get_weather(city:str, day: str)->str:
        """
            Get weather based on city
            Args:
                city: City which you want query for weather
        """
        return "23 degrees celcius"


# level - 2
# Access Short term memory access state

from langchain.tools import ToolRuntime

def tool3():

    @tool
    def last_message(runtime:  ToolRuntime):
        messages = runtime.state["messages"]
        return messages[-1]

    @tool
    def user_pref(name : str, runtime: ToolRuntime):
        preferences = runtime.state.get("user_preferences", {})
        return preferences.get("location", "knr")


# level - 3
# update the state
from langchain.agents import AgentState, create_agent
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.chat_models import init_chat_model

def tool4():

    class CustomModel(AgentState):
        name:str

    @tool
    def change_username(new_name: str, runtime: ToolRuntime[None, CustomModel]):
        """This function is used to change the user name"""
        return Command(
            update={
                "user_name": new_name,
                "messages": [
                    ToolMessage(
                        content=f"Username changed to {new_name}", 
                        tool_call_id=runtime.tool_call_id 
                    )
                ]
            }
        )

    model = init_chat_model("ollama:llama3.2")
    agent = create_agent(
        model=model,
        tools=[change_username],
        state_schema=CustomModel
    )

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Your username is pavaneeshwar"
            },
            {
                "role": "user",
                "content": "Change your username to chandana"
            },
            {
                "role": "user",
                "content": "what is the username"
            }
        ]
    })
    print(response["messages"][-1].content)



# level - 4
# context
from dataclasses import dataclass

def tool5():

    database = [
        {
            "user_id": "12345",
            "balance": 200
        },
        {
            "user_id": "6789",
            "balance": 800
        }
    ]

    @dataclass
    class UserContext:
        user_id: str


    @tool(description="Used when a user asks to fetch account balance")
    def get_balance(runtime: ToolRuntime[UserContext]):
        """Balance tool that gives the balance based on user context and return balance if exists"""
        for record in database:
            if record["user_id"] == runtime.context.user_id:
                return record["balance"]
        return "User with this user id doesn't exists in the database"

    model = init_chat_model("ollama:llama3.2")
    agent = create_agent(
        model=model,
        tools=[get_balance],
        context_schema=UserContext
    )   

    response = agent.invoke({
        "messages": {
            "role": "user",
            "content": "what is the balance ?"
        }
    }, context=UserContext(user_id="12345"))

    print(response["messages"][-1].content)


# level - 5
# Long term memory - use Postgres or Redis or MongoDB Store for production

from langgraph.store.memory import InMemoryStore
def tool6():

    @tool(description="Used to save user info")
    def save_user(user_id: str, username:str, runtime: ToolRuntime):
        """
            Used to save user in the memory store
            Args:
                user_id: An id assigned for the user
                username: Username alloted to the user
        """
        store = runtime.store
        store.put(("users"), user_id, username)
        return "Succesfully saved username"

    @tool(description="Which gets all the users in the store")
    def get_all_users(runtime: ToolRuntime):
        """
            Used when user asks to fetch all the users in the database
        """
        store = runtime.store
        users = store.search(("users"))
        return [user.value for user in users]

    store = InMemoryStore()

    model = init_chat_model("ollama:llama3.2")
    agent = create_agent(
        model=model,
        tools=[save_user, get_all_users],
        store=store
    )  

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Save user with user_id 123 and username pavaneeshwar"
            },
            {
                "role": "user",
                "content": "Save user with user_id 567 and username chandana"
            }
        ]
    })

    print(response["messages"][-1].content)

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Give all the users in the database"
            }
        ]
    })

    print(response["messages"][-1].content)


# level - 6
# stream writer (used when we want to stream execution info ), execution info, server info
def tool7():

    @tool
    def get_weather(city:str, runtime: ToolRuntime):
        writer = runtime.stream_writer
        writer(f"Looking for city {city}")

        info = runtime.execution_info # thread id, run id, etc
        print(info)

        server_info = runtime.server_info # assistant id, graph id ,etc
        print(server_info)



# level - 7
# Return types ( we have seen before like string, object, command)
def tool8():

    @tool(return_direct=True) # this doesn't make a new model call, just return from this function
    def get_weather(city:str):
        pass


from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
# level - 8
# handle exceptions and convert them as tool messages
def tool9():

    @wrap_tool_call 
    def get_weather(request: ToolCallRequest, handler) -> ToolMessage:
        try:
            return handler(request)
        except Exception:
            return ToolMessage("something went wrong", tool_call_id=request.tool_call["id"])


# level - 9 
# Dynamic tools

def tool10():

    @wrap_tool_call
    def decide_which_tools(request: ToolCallRequest, handler):
        tools = request.tools
        request.override(tools=tools) # you can modify which tools you want
        handler(request)

        state = request.runtime.state # either we can use state to filter out required tools
        context = request.runtime.context # or context


