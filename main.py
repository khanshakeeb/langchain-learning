from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer:str = Field(description="The agent's answer to the question")
    sources:List[Source] = Field(description="List of sources used to generate the question", default_factory=list)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print(os.getenv("TAVILY_API_KEY"))
    print("Search Agent")
    response = agent.invoke({"messages": HumanMessage(content="Search for 3 AI Enginer job posting for langchain in linkedin Cologne, Germany")})
    print(response)

if __name__ == "__main__":
    main()
