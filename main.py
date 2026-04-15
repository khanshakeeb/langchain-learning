from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

# tavily_client = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool the search over the web for the given query
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for {query}")
#     return tavily_client.search(query=query)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print(os.getenv("TAVILY_API_KEY"))
    print("Search Agent")
    response = agent.invoke({"messages": HumanMessage(content="Search for 3 AI Enginer job posting for langchain in linkedin Cologne, Germany")})
    print(response)

if __name__ == "__main__":
    main()
