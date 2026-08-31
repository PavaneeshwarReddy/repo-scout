"""
Tools:
- By default functions docstring defines the model when to use it
- Args schema can be passed
- Tools can access run time information from ToolRuntime

Access Context
As tools are very powerful but when they are able to shared context like messages, file systems, etc

1. Short term Memory: only exisits until the conversation ends



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
# Short term memory access ( access context )
def tool3():
    pass