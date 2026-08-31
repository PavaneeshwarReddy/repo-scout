

"""
Models:
They can perform any of the task given below
    - Tools calling
    - Structured output
    - Multi-model capabilities
    - Reasoning

They can be used either in an agentic loop or outside the loop also ( basic execution without the need to agent)

Model can be invoked in either of 3 ways
- Invoke -> basic method which is called out of agentic loop and we can pass conversation history
- Stream -> stream output while generating one by one instead of waiting for all
- Batch -> run multiple independent queries parallely using this

Tool calls:
- Manually invoke the tool
- Force to choose a tool
- Parallel execution of tool when needed decided by the model itself
- Stream tool calls 

Messages & Roles:

Messages: HumanMessage , AIMessage

Roles:
system    : instructions for the AI
user      : request from the user
assistant : previous response from the AI
tool      : result returned by a tool

"""

from langchain.chat_models import init_chat_model

# level - 0
# intialize a init chat model
def model1():
    # basic model decleration
    model = init_chat_model(
        "ollama:llama3.2"
    )

    response = model.invoke("What you can do ?").content

# level - 1
# knowing all the parameters
def model2():
    model = init_chat_model(
       "ollama:llama3.2",
       api_key="", # if you are some third party provider other than local
       temperature=0.2, # lower (tools) the value it is deterministic and higher (conversational responses) the value more creative
       max_tokens=2000, # limit the max tokens in the response
       timeout=300, # timeout for responses
       max_retries=2, # incase of model not reponding how many times to retry before quit
    )



from langchain.messages import HumanMessage, AIMessage

# level - 2
# pass conversation history to a model and use invoke
def model3():
    conversation1= [
        {"role": "user", "content": "where am i located in ? "},
        {"role": "assistant", "content": "japan"},
        {"role": "user", "content": "can you translate this , Hi what's your name ?"}
    ]

    conversation2 = [
        HumanMessage("where am i located in ?"),
        AIMessage("japan"),
        HumanMessage("can you translate this, Hi what's your name ? ")
    ]

    model = init_chat_model(
        "ollama:llama3.2",
    )
    response = model.invoke(conversation1).content
    print(response)
    response = model.invoke(conversation2).content


# level - 3
# stream the output
def model4():
    model = init_chat_model(
        "ollama:llama3.2",
    )

    stream = model.stream("can you write code for binary search with time complexity analysis in python and go ? ")
    for chunk in stream:
        print(chunk.content)


# level - 4
# batch the input
def model5():
    model = init_chat_model(
        "ollama:llama3.2",
    )

    # this only returns when all the batches are returned with a response 
    responses = model.batch([
        "translate i love you in japaneese",
        "translate i love you in spanish", 
        "trnslate somethiing is fishy in telugu"
    ], config={
        "max_concurrency": 3 # max concurrent calls to the models
    })

    for response in responses:
        print(response.content)

    # this returns when a task is completed in a batch
    respones = model.batch_as_completed([
        "translate i love you in japaneese",
        "translate i love you in spanish", 
        "trnslate somethiing is fishy in telugu"
    ])

    for response in responses:
        print(response.content)


from langchain.tools import tool

# level - 5
# tool calls and bind tools with a model and invoke
def model6():

    @tool
    def get_weather(city : str) -> str:
        """Gives the weather of the city provided"""
        return "23 degrees celcius"

    model = init_chat_model(
        "ollama:llama3.2",
    )

    model_with_tools = model.bind_tools([get_weather])
    responses = model_with_tools.invoke("Whats the weather in nyc and calfornia ? ")
    for tool_call in responses.tool_calls:
        print(tool_call["name"], tool_call["args"])


# level - 6
# tool calls are not invoked automatically when they are binded to model, we need to invoke and give results
# agent automatically does for you when u use create_agent
def model7():

    @tool
    def get_weather(city : str) -> str:
        """Gives the weather of the city provided"""
        return "23 degree"

    model = init_chat_model(
        "ollama:llama3.2",
    )

    # you are just saying your model that it contains some tools
    model_with_tools = model.bind_tools([get_weather], tool_choice='get_weather') # you can force model to choose this tool
    messages = [{"role": "user", "content": "What's the weather in Boston?"}]

    # none of the tool is executed it just decides the tools in aiMessages
    airesponse = model_with_tools.invoke(messages)
    messages.append(airesponse)

    # make the tool call and append the result
    for tool_call in airesponse.tool_calls:
        ai_tool_res = get_weather.invoke(tool_call)
        messages.append(ai_tool_res)

    # at last again invoke with all the message history
    result = model.invoke(messages)
    print(result.content)


# level - 7
# stream the tool calls
def model8():

    @tool
    def get_weather(city : str) -> str:
        """Gives the weather of the city provided"""
        return "23 degree"

    model = init_chat_model(
        "ollama:llama3.2",
    )

    # you are just saying your model that it contains some tools
    model_with_tools = model.bind_tools([get_weather], tool_choice='get_weather') # you can force model to choose this tool
    messages = [{"role": "user", "content": "What's the weather in Boston?"}]

    # none of the tool is executed it just decides the tools in aiMessages
    for chunk in model_with_tools.stream(messages):
        for tool_chunk in chunk.tool_call_chunks:
            print(tool_chunk)


# level - 8
# Structured output from a model, we can also json 
from pydantic import BaseModel, Field

def model9():

    class CodeResponse(BaseModel):
        code: str = Field(description="Contains real executable code that is asked in the query ")
        language: str = Field(description="contains language in which code is written")
        time_cmplx: str = Field(description="Time complexity of the code written")
        space_cmplx: str =Field(description="Space complexity of the code written")

    model = init_chat_model(
        "ollama:llama3.2",
    )

    structured_model = model.with_structured_output(CodeResponse)

    response = structured_model.invoke("Can you give me the code for binary search ? " )
    print(response)


# level - 9 
# invoking a multimodel
def model10():
    model = init_chat_model(
        "ollama:llama3.2",
    )

    responses = model.invoke("Create an image of a cat")
    print(responses.content_blocks)


# level-10
# reasoning -> some models reason to arrive at a final answer
def model11():
    model = init_chat_model(
        "ollama:llama3.2",
    )

    responses = model.invoke("Can your reason why birds in that way ?", reasoning_effot="high") # this has to be supported by the provider
    for response in responses.content_blocks:
        if response.get("type") == "reasoning":
            print(response)


# level-11
# prompt caching -> this sometimes implicitly supported by provider
# Langchain middlware support caching as middlewares for anthropic and open ai


# level - 12
# rate limiting
from langchain.rate_limiters import InMemoryRateLimiter

def model12():
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=10,
        check_every_n_seconds=10,
        max_bucket_size=10
    )

    model = init_chat_model(
        "ollama:llama3.2",
        rate_limiter=rate_limiter
    )


from langchain_core.callbacks import UsageMetadataCallbackHandler

# level - 13
# checking token usages, with a call back that collects all token usages
def model13():
    call_back = UsageMetadataCallbackHandler()
    model = init_chat_model(
        "ollama:llama3.2",
    )

    model.invoke("write code for pythgon in binary search ", config={
        "callbacks":[call_back],
        "run_name": "binary_search_run", # you can add other which alters the behaviour of model
    })

    model.invoke("write code for pythgon in bfs  ", config={
        "callbacks":[call_back]
    })

    print(call_back.usage_metadata)


# level - 14
# configurable model and dynamic configurable model
def model14():

    # based on some condition we choose different models to perform the task
    configurable_model = init_chat_model()
    condition = True
    if condition:
        configurable_model.invoke(config={
            "configurable": {
                "model":  "ollama:llama3.2"
            }
        })


    # dynamically configurable model based on message count
    base_model = init_chat_model("ollama:llama3.2")
    advanced_model = init_chat_model("ollama:llama3.2")

    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        messages = request.state["messages"]
        if len(messages) > 10:
            model=advanced_model
        else:
            model=base_model

        return handler(request.override(model=model))

    create_agent(
        model=base_model,
        middleware=[dynamic_model_selection]
    )



