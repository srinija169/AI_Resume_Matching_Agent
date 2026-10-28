"""LangGraph-powered conversational resume matching agent."""
from typing import Any
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from requirements import extract_requirements, merge_requirements
from rag_search import ResumeRetriever
from ranking import rank_candidates
from tools import compare_candidates, generate_interview_questions

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    job_description: str
    requirements: dict[str, Any]
    candidate_pool: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    ranking_history: list[dict[str, Any]]
    screening_round: str
    feedback: str
    report: str
    action: str
    comparison: list[dict[str, Any]]
    interview_questions: dict[str, list[str]]
    previous_shortlist: list[dict[str, Any]]
    awaiting_feedback: bool
    feedback_prompt: str

retriever = ResumeRetriever()

def _last_user_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return state.get("job_description", "")




def _intent(text: str) -> str:
    t = text.lower()

    # Candidate-specific screening must take priority over a general deep screen.
    import re
    has_candidate_id = bool(re.search(r"\bC\d{3}\b", text, re.I))
    is_deep_screen = any(x in t for x in (
        "deep screen", "deep-screen", "final round",
        "hire/no-hire", "deep analysis", "final recommendation"
    ))

    if is_deep_screen and has_candidate_id:
        return "candidate_deep_screen"

    if is_deep_screen:
        return "deep_screen"

    if any(x in t for x in (
        "compare", "side by side", "head-to-head"
    )):
        return "compare"

    if any(x in t for x in (
        "interview question", "screening question",
        "prepare questions"
    )):
        return "questions"

    if any(x in t for x in (
        "explain why", "why did", "why was",
        "explain the ranking", "why rank",
        "ranked higher", "ranking explanation",
        "top candidate"
    )):
        return "explain"

    if any(x in t for x in (
        "make ", "now require", "mandatory", "must-have",
        "must have", "rerank", "re-rank",
        "change the requirement", "at least"
    )):
        return "refine"

    return "search"


def parse_request(state: AgentState) -> dict:
    text = _last_user_text(state)
    action = _intent(text)
    # A refinement uses existing criteria and appends the latest instruction.
    jd = state.get("job_description", "")
    if action == "search" or not jd:
        jd = text
    return {"action": action, "job_description": jd, "feedback": text}



def extract_node(state: AgentState) -> dict:
    action = state.get("action", "search")
    current = state.get("requirements") or {}
    text = state.get("feedback", "")

    # A new job-search request must establish fresh criteria,
    # even if the request also mentions deep screening.
    if action == "search":
        jd = state.get("job_description", "")
        return {"requirements": extract_requirements(jd)}

    # Refinements update the existing criteria.
    if action == "refine":
        extracted = extract_requirements(text)
        return {
            "requirements": merge_requirements(current, extracted)
        }

    # Follow-up actions must retain the active criteria.
    if action in {
        "compare", "explain", "questions",
        "deep_screen", "candidate_deep_screen"
    }:
        if current:
            return {"requirements": current}

        # Recover criteria from the saved job description if missing.
        jd = state.get("job_description", "")
        if jd:
            return {"requirements": extract_requirements(jd)}

        return {"requirements": {}}

    return {
        "requirements": extract_requirements(
            state.get("job_description", "")
        )
    }


def search_node(state: AgentState) -> dict:
    action = state.get("action")

    if action in {
        "compare", "explain", "questions",
        "deep_screen", "candidate_deep_screen"
    } and state.get("candidate_pool"):
        return {}

    query = (
        state.get("job_description", "")
        + " "
        + str(state.get("requirements", {}))
    )
    results = retriever.search(query, limit=100)
    return {"candidate_pool": results}



def rank_node(state: AgentState) -> dict:
    action = state.get("action", "search")
    old_shortlist = state.get("shortlist", [])

    # These actions should use the existing ranking unchanged.
    if action in {"compare", "explain", "questions"}:
        return {}

    # Deep screening analyzes the existing top 10.
    # It must not discard or replace the active job requirements.
    if action == "deep_screen" and old_shortlist:
        ranked = old_shortlist[:10]
        history = list(state.get("ranking_history", []))
        history.append({
            "action": action,
            "round": "deep_analysis",
            "criteria": state.get("requirements", {}),
            "previous_top_ids": [
                c.get("candidate_id") for c in old_shortlist
            ],
            "new_top_ids": [
                c.get("candidate_id") for c in ranked
            ],
        })

        return {
            "previous_shortlist": old_shortlist,
            "shortlist": ranked,
            "ranking_history": history,
            "screening_round": "deep_analysis",
        }

    
    if action == "candidate_deep_screen":
        import re
        text = state.get("feedback", "")
        match = re.search(r"\bC\d{3}\b", text, re.I)
        candidate_id = match.group(0).upper() if match else ""

        candidate = next(
            (
                c for c in old_shortlist
                if c.get("candidate_id", "").upper() == candidate_id
            ),
            None,
        )

        if candidate is None:
            return {
                "shortlist": [],
                "screening_round": "candidate_deep_analysis",
            }

        return {
            "shortlist": [candidate],
            "screening_round": "candidate_deep_analysis",
        }

    # Initial searches and explicit refinements run a fresh ranking.
    pool = state.get("candidate_pool", [])
    ranked = rank_candidates(
        pool,
        state.get("requirements", {}),
        limit=10,
    )

    history = list(state.get("ranking_history", []))
    history.append({
        "action": action,
        "round": "initial",
        "criteria": state.get("requirements", {}),
        "previous_top_ids": [
            c.get("candidate_id") for c in old_shortlist
        ],
        "new_top_ids": [
            c.get("candidate_id") for c in ranked
        ],
    })

    return {
        "previous_shortlist": old_shortlist,
        "shortlist": ranked,
        "ranking_history": history,
        "screening_round": "initial",
    }


def respond_node(state: AgentState) -> dict:
    action = state.get("action", "search")
    shortlist = state.get("shortlist", [])
    reqs = state.get("requirements", {})
    text = state.get("feedback", "")
    report_lines = []

    if action == "compare":
        ids = [c["candidate_id"] for c in shortlist if c["candidate_id"].lower() in text.lower()]
        if not ids:
            # If the user says "top 3", compare top three currently ranked candidates.
            ids = [c["candidate_id"] for c in shortlist[:3]]
        comparison = compare_candidates(ids, shortlist)
        report_lines.append("### Candidate comparison")
        for c in comparison:
            report_lines.append(f"- **{c['name']} ({c['candidate_id']})** — score {c['score']}/100; experience {c['experience_years']} years; matched skills: {', '.join(c['matched_skills']) or 'none'}; gaps: {'; '.join(c['gaps'])}")
        report = "\n".join(report_lines)
        return {"comparison": comparison, "report": report, "messages": [AIMessage(content=report)]}

    if action == "questions":
        import re
        ids = re.findall(r"C\d{3}", text.upper())
        candidate_id = ids[0] if ids else (shortlist[0]["candidate_id"] if shortlist else "")
        questions = generate_interview_questions(candidate_id, shortlist, reqs)
        report = "### Suggested screening questions\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        return {"interview_questions": questions, "report": report, "messages": [AIMessage(content=report)]}
 
    if action == "explain":
        # Explain the current ranking without recalculating scores.
        names = [
            c for c in shortlist
            if c.get("name", "").lower() in text.lower()
            or c.get("candidate_id", "").lower() in text.lower()
        ]
        selected = names[:2] if names else shortlist[:2]

        report_lines.append("### Ranking explanation")

        if not selected:
            report = "No candidates are available to explain yet. Run a search first."
            return {
                "report": report,
                "messages": [AIMessage(content=report)],
            }

        for c in selected:
            report_lines.append(
                f"**{c['name']} ({c['candidate_id']})** — "
                f"{c['score']}/100; "
                f"must-have pass: {'Yes' if c['must_have_pass'] else 'No'}."
            )
            report_lines.append(
                "**Matching skills:** "
                + (", ".join(c.get("matched_skills", [])) or "None listed")
            )
            report_lines.append(
                "**Evidence:** " + "; ".join(c.get("evidence", [])[:3])
            )
            report_lines.append(
                "**Gaps:** " + "; ".join(c.get("gaps", []))
            )

        if len(selected) >= 2:
            first, second = selected[0], selected[1]
            score_difference = round(
                first.get("score", 0) - second.get("score", 0), 1
            )

            report_lines.append(
                f"**Why the first candidate ranks higher:** "
                f"{first['name']} scores {score_difference} points "
                f"{'above' if score_difference >= 0 else 'below'} "
                f"{second['name']}."
            )

            if first.get("must_have_pass") != second.get("must_have_pass"):
                report_lines.append(
                    "Must-have criteria also affect the ordering: "
                    f"{first['name']} "
                    f"{'passes' if first['must_have_pass'] else 'does not pass'} "
                    "the must-have criteria, while "
                    f"{second['name']} "
                    f"{'passes' if second['must_have_pass'] else 'does not pass'}."
                )

            first_skills = set(first.get("matched_skills", []))
            second_skills = set(second.get("matched_skills", []))
            additional = sorted(first_skills - second_skills)

            if additional:
                report_lines.append(
                    f"Skills evidenced for {first['name']} but not "
                    f"listed as matched for {second['name']}: "
                    + ", ".join(additional)
                    + "."
                )

            report_lines.append(
                "The comparison is based on the supplied resume data and "
                "configured scoring rules, not an independent verification "
                "of either candidate's abilities."
            )

        report = "\n".join(report_lines)
        return {
            "report": report,
            "messages": [AIMessage(content=report)],
        }

    report_lines.append(f"### {state.get('screening_round', 'initial').replace('_', ' ').title()} results")
    report_lines.append("**Must-have criteria:** " + ", ".join(reqs.get("must_have", {}).get("skills", [])) +
                        f"; minimum experience: {reqs.get('must_have', {}).get('minimum_experience_years', 0)} years.")
    report_lines.append("**Nice-to-have criteria:** " + ", ".join(reqs.get("nice_to_have", {}).get("skills", [])))
    for i, c in enumerate(shortlist, 1):
        report_lines.append(f"#### {i}. {c['name']} ({c['candidate_id']}) — {c['score']}/100")
        report_lines.append(f"- Must-have pass: **{'Yes' if c['must_have_pass'] else 'No / gap found'}**")
        report_lines.append("- Strengths: " + "; ".join(c["strengths"]))
        report_lines.append("- Gaps: " + "; ".join(c["gaps"]))
        report_lines.append("- Evidence: " + "; ".join(c["evidence"][:3]))
    if action == "refine":
        report_lines.append("\n### Ranking changes")
        previous = state.get("previous_shortlist", [])
        report_lines.append("Previous top candidates: " + ", ".join(c["name"] for c in previous[:3]))
        report_lines.append("Updated top candidates: " + ", ".join(c["name"] for c in shortlist[:3]))
        report_lines.append("Scores and order are recalculated using the updated criteria.")
    
        
    if action in {"deep_screen", "candidate_deep_screen"}:

        # Add evidence-based screening recommendations.
        report_lines.append("\n### Provisional recommendation")
        report_lines.append("### Final screening recommendation")   
        report_lines.append(
            "These are screening recommendations, not automatic hiring decisions."
        )

        qualified = [
            c for c in shortlist
            if c.get("must_have_pass", False)
        ]

        if qualified:
            best = qualified[0]
            report_lines.append(
                f"\n**Top candidate for further consideration:** "
                f"{best['name']} ({best['candidate_id']})"
            )
            report_lines.append(
                f"- Screening score: {best['score']}/100"
            )
            report_lines.append(
                "- Status: Eligible for further consideration; "
                "interview validation is still required."
            )
            report_lines.append(
                "- Supporting evidence: "
                + "; ".join(best.get("evidence", [])[:3])
            )
        else:
            report_lines.append(
                "\n**Recommendation:** No candidate in the shortlist "
                "currently passes all must-have criteria."
            )
            report_lines.append(
                "Review the requirements and candidate evidence before "
                "deciding whether to proceed."
            )

        report_lines.append("\n### Candidate-by-candidate decisions")

        for candidate in shortlist:
            name = candidate.get("name", "Unknown candidate")
            candidate_id = candidate.get("candidate_id", "Unknown ID")
            gaps = candidate.get("gaps", [])

            if candidate.get("must_have_pass", False):
                status = "Eligible for further consideration"
                explanation = (
                    "Passes the configured must-have checks; "
                    "verify qualifications during the interview."
                )
            else:
                status = "Needs review"
                explanation = (
                    "One or more must-have checks are not satisfied "
                    "or have identified gaps."
                )

            report_lines.append(
                f"\n**{name} ({candidate_id}) — {status}**"
            )
            report_lines.append(f"- Score: {candidate.get('score', 0)}/100")
            report_lines.append(f"- Reason: {explanation}")
            report_lines.append(
                "- Gaps to review: " + ("; ".join(gaps) if gaps else "None recorded")
            )

        report_lines.append(
            "\n**Human decision required:** Do not treat this ranking "
            "as a final hire/no-hire decision. Validate skills, experience, "
            "and role fit through structured interviews and appropriate "
            "human review."
        )

        report_lines.append("\n### Suggested interview questions")

        # The deep-screening round keeps the top 3 candidates.
        interview_sections = []
        questions_by_candidate = {}

        for candidate in shortlist:
            candidate_id = candidate.get("candidate_id", "")

            if not candidate_id:
                continue

            questions = generate_interview_questions(
                candidate_id, shortlist, reqs
            )

            questions_by_candidate[candidate_id] = questions
            interview_sections.append(
                f"#### {candidate.get('name', candidate_id)} "
                f"({candidate_id})\n"
                + "\n".join(
                    f"{i}. {question}"
                    for i, question in enumerate(questions, 1)
                )
            )

        if interview_sections:
            report_lines.extend(interview_sections)
        else:
            report_lines.append("No interview questions could be generated.")

        report = "\n".join(report_lines)

        return {
            "report": report,
            "interview_questions": questions_by_candidate,
            "messages": [AIMessage(content=report)],
        }

    report = "\n".join(report_lines)
    return {"report": report, "messages": [AIMessage(content=report)]}

def human_feedback_node(state: AgentState) -> dict:
    """Offer the user a clear opportunity to refine or accept the results."""
    action = state.get("action", "search")

    # Follow-up actions have already supplied feedback in this turn.
    # Keep their existing responses unchanged.
    if action in {"refine", "compare", "explain", "questions", "deep_screen"}:
        return {
            "awaiting_feedback": False,
            "feedback_prompt": "",
        }

    prompt = (
        "\n\n### Human feedback\n"
        "Would you like to refine the criteria and re-rank the candidates, "
        "compare candidates, ask why someone ranked higher, or run a deep screen? "
        "For example: `Now require TypeScript as a must-have.`"
    )

    report = state.get("report", "")
    updated_report = report + prompt

    return {
        "awaiting_feedback": True,
        "feedback_prompt": prompt,
        "report": updated_report,
        "messages": [AIMessage(content=prompt)],
    }

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_request", parse_request)
    builder.add_node("extract_requirements", extract_node)
    builder.add_node("search_resumes", search_node)
    builder.add_node("rank_candidates", rank_node)
    builder.add_node("generate_report", respond_node)
    builder.add_node("human_feedback", human_feedback_node)

    builder.add_edge(START, "parse_request")
    builder.add_edge("parse_request", "extract_requirements")
    builder.add_edge("extract_requirements", "search_resumes")
    builder.add_edge("search_resumes", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", "human_feedback")
    builder.add_edge("human_feedback", END)

    return builder.compile(checkpointer=InMemorySaver())

def invoke_agent(graph, text: str, thread_id: str = "default", previous_state: dict | None = None) -> dict:
    """Invoke graph for one turn; the checkpointer preserves graph state per thread."""
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )
