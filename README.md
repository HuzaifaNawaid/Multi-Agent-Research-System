# 🔬 ResearchMind — Multi-Agent AI Research Pipeline

> Four specialized AI agents collaborate to search, scrape, write, and critique a polished research report on any topic — powered by LangGraph and OpenAI.

---

## Overview

**ResearchMind** is an autonomous, multi-agent AI research system built on [LangGraph](https://github.com/langchain-ai/langgraph). Given any research topic, it orchestrates four specialized agents in a sequential pipeline:

1. **Searches** the web for recent and reliable information
2. **Scrapes** the most relevant source for deeper content
3. **Writes** a structured, professional research report
4. **Critiques** the report with a score and actionable feedback

The system exposes two interfaces: a command-line pipeline (`pipeline.py`) and a Streamlit web app (`app.py`).

---

## Architecture

```
User Input (Topic)
        │
        ▼
┌───────────────┐
│  Search Agent │  ← LangGraph ReAct Agent + Tavily web_search tool
└──────┬────────┘
       │  search_results
       ▼
┌───────────────┐
│  Reader Agent │  ← LangGraph ReAct Agent + scrape_url tool (BeautifulSoup)
└──────┬────────┘
       │  scraped_content
       ▼
┌───────────────┐
│  Writer Chain │  ← LangChain Prompt | GPT-4o-mini | StrOutputParser
└──────┬────────┘
       │  report
       ▼
┌───────────────┐
│  Critic Chain │  ← LangChain Prompt | GPT-4o-mini | StrOutputParser
└──────┬────────┘
       │  feedback
       ▼
  Final State Dict
```

The entire pipeline is managed as a **LangGraph StateGraph**, where each node reads from and writes to a shared `ResearchState` TypedDict.

---

## Project Structure

```
.
├── agents.py          # Core agents, chains, graph nodes, and graph builder
├── tools.py           # LangChain tools: web_search (Tavily) and scrape_url
├── pipeline.py        # CLI entrypoint — invokes the graph and prints results
├── app.py             # Streamlit web UI with live pipeline status display
└── requirements.txt   # All Python dependencies
```

---

## Agent Pipeline

### 1 · Search Agent

- **Type:** LangGraph ReAct Agent
- **Tool:** `web_search` — queries Tavily Search API, returns up to 5 results with titles, URLs, and content snippets (300 chars each)
- **Output:** `search_results` — formatted string of search result snippets

### 2 · Reader Agent

- **Type:** LangGraph ReAct Agent
- **Tool:** `scrape_url` — fetches a URL, strips scripts/styles/nav/footer with BeautifulSoup, and returns up to 3,000 characters of clean text
- **Input:** Receives the first 800 characters of `search_results` and picks the most relevant URL to scrape
- **Output:** `scraped_content` — raw extracted article/page text

### 3 · Writer Chain

- **Type:** LangChain LCEL chain (`prompt | llm | StrOutputParser`)
- **Model:** `gpt-4o-mini` (temperature 0)
- **Input:** Combined `search_results` + `scraped_content`
- **Output:** `report` — a structured Markdown report with Introduction, Key Findings (3+), Conclusion, and Sources

### 4 · Critic Chain

- **Type:** LangChain LCEL chain (`prompt | llm | StrOutputParser`)
- **Model:** `gpt-4o-mini` (temperature 0)
- **Input:** The generated `report`
- **Output:** `feedback` — a structured critique with Score (X/10), Strengths, Areas to Improve, and a one-line verdict

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) >= 0.2.0 |
| LLM | OpenAI `gpt-4o-mini` via `langchain-openai` |
| Web Search | [Tavily](https://tavily.com/) Python client |
| Web Scraping | `requests` + `BeautifulSoup4` |
| Prompt/Chain Building | LangChain Core (LCEL) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Environment Config | `python-dotenv` |
| Logging/Display | `rich` |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/researchmind.git
cd researchmind

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

The app loads these automatically via `python-dotenv`.

---

> Made with ❤️ by [Huzaifa Nawaid](https://github.com/HuzaifaNawaid)

> Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://www.langchain.com/) · [Streamlit](https://streamlit.io/) · [Tavily](https://tavily.com/)
