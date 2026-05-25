from graph import build_research_graph


def run_research_pipeline(topic: str) -> dict:

    graph = build_research_graph()

    print("\n" + "=" * 50)
    print(" Starting LangGraph Research Pipeline")
    print("=" * 50)

    initial_state = {"topic": topic}
    final_state = graph.invoke(initial_state)

    print("\n[Search Results]\n",   final_state.get("search_results", ""))
    print("\n[Scraped Content]\n",  final_state.get("scraped_content", ""))
    print("\n[Final Report]\n",     final_state.get("report", ""))
    print("\n[Critic Feedback]\n",  final_state.get("feedback", ""))

    return final_state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)
