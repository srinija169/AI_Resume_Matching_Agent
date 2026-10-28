"""Lightweight integration checks for five conversation intents."""
from langchain_core.messages import HumanMessage
from matching_agent import build_graph

def ask(graph, text, thread="test-thread"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread}},
    )

def test_initial_search():
    graph = build_graph()
    result = ask(graph, "Find React developers with 3+ years experience", "search")
    assert result["shortlist"]
    assert "results" in result["report"].lower()

def test_compare_top_three():
    graph = build_graph()
    ask(graph, "Find React developers with 3+ years experience", "compare")
    result = ask(graph, "Compare the top 3 matches side by side", "compare")
    assert "comparison" in result["report"].lower()

def test_ranking_explanation():
    graph = build_graph()
    ask(graph, "Find React developers with 3+ years experience", "explain")
    result = ask(graph, "Why did Alex Morgan rank higher than Jamie Patel?", "explain")
    assert "ranking explanation" in result["report"].lower()

def test_refinement():
    graph = build_graph()
    ask(graph, "Find React developers with 3+ years experience", "refine")
    result = ask(graph, "Make TypeScript mandatory and rerank everyone", "refine")
    assert "typescript" in result["requirements"]["must_have"]["skills"]
    assert "ranking changes" in result["report"].lower()

def test_deep_screen():
    graph = build_graph()
    ask(graph, "Find React developers with 3+ years experience", "deep")
    result = ask(graph, "Deep-screen the top 10 and prepare interview questions", "deep")
    assert result["screening_round"] == "deep_analysis"
    assert "provisional recommendation" in result["report"].lower()

def test_deep_screen():
    graph = build_graph()
    ask(
        graph,
        "Find React developers with 3+ years experience",
        "deep",
    )
    result = ask(
        graph,
        "Deep-screen the top 10 and prepare interview questions",
        "deep",
    )

    assert result["screening_round"] == "deep_analysis"
    assert "provisional recommendation" in result["report"].lower()
    assert len(result["shortlist"]) == 10
    assert len(result["interview_questions"]) == 10
    assert all(result["interview_questions"].values())
