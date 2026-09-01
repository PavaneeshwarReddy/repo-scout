"""
Structured output:
Declared how a model outcome is returned, we can either use pydantic or json schema
Validations on the output can be done using pydantic or json schema

ToolStrategy: human -> model -> tool call for better response structure
ProviderStrategy: same as above but model itself has a output response format

"""

# level - 0
# make structured output as a tool call
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel, Field
from typing import Union

def so1():

    class WeatherModel(BaseModel):
        city:str = Field(description="City name as supplied")
        temperature: str = Field(description="Weather which you had as an input")

    agent = create_agent(
        "ollama:llama3.2",
        response_format=Union[
            ToolStrategy(schema=WeatherModel,
                         tool_message_content="Weather fetched successfully", 
                         description="Weather information",
                         handle_errors=(TypeError, ValueError) # handler_errors = "can be simple message"
                         ), 
            ProviderStrategy(WeatherModel)
        ]
    )



