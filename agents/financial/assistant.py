from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name="web_search_agent",
    role="search the web for the information",
    provider=Groq(
        api_key=groq_api_key,
        id="llama3-groq-70b-8192-tool-use-preview"
    ),
    tools=[
        DuckDuckGo()
    ],
    instructions=[
        "Always include the source of the information",
    ],
    show_tool_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="financial_agent",
    provider=Groq(
        api_key=groq_api_key,
        id="llama3-groq-70b-8192-tool-use-preview"
    ),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True,
                      stock_fundamentals=True, company_news=True)
    ],
    instructions=[
        "Use tables to display the information",
    ],
    show_tool_calls=True,
    markdown=True
)

multimodal_ai_agent = Agent(
    provider=Groq(id="llama-3.1-70b-versatile"),
    team=[web_search_agent, financial_agent],
    instructions=[
        "Always include sources of the information",
        "Use tables to display the information"
    ],
    show_tool_calls=True,
    markdown=True
)

multimodal_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA",
    stream=True
)
