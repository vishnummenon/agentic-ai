from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(
    agents=[web_search_agent, financial_agent],
).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)
