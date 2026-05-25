from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Shared state schema ───────────────────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str

# ── Agents (LangGraph react agents) ──────────────────────────────────────────
search_agent = create_react_agent(model=llm, tools=[web_search])
reader_agent = create_react_agent(model=llm, tools=[scrape_url])

# ── Writer chain ──────────────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()

# ── Critic chain ──────────────────────────────────────────────────────────────
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])
critic_chain = critic_prompt | llm | StrOutputParser()

# ── Graph nodes ───────────────────────────────────────────────────────────────
def search_node(state: ResearchState) -> ResearchState:
    result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {state['topic']}")]
    })
    return {"search_results": result["messages"][-1].content}


def reader_node(state: ResearchState) -> ResearchState:
    result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{state['topic']}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    return {"scraped_content": result["messages"][-1].content}


def writer_node(state: ResearchState) -> ResearchState:
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    report = writer_chain.invoke({
        "topic": state["topic"],
        "research": research_combined,
    })
    return {"report": report}


def critic_node(state: ResearchState) -> ResearchState:
    feedback = critic_chain.invoke({"report": state["report"]})
    return {"feedback": feedback}


# ── Build graph ───────────────────────────────────────────────────────────────
def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "reader")
    graph.add_edge("reader", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
