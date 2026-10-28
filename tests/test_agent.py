from requirements import extract_requirements, merge_requirements
from ranking import rank_candidates, compare_candidates, generate_interview_questions
from rag_search import ResumeRetriever, load_resumes

def test_extract_react_and_years():
    req = extract_requirements("Find React developers with 3+ years experience")
    assert "react" in req["all_requested_skills"]
    assert req["must_have"]["minimum_experience_years"] == 3

def test_mandatory_refinement():
    current = extract_requirements("Find React developers with 3+ years experience")
    update = extract_requirements("Make TypeScript mandatory and rerank everyone")
    merged = merge_requirements(current, update)
    assert "typescript" in merged["must_have"]["skills"]

def test_retriever_returns_candidates():
    retriever = ResumeRetriever()
    results = retriever.search("React TypeScript", limit=5)
    assert len(results) == 5
    assert all("candidate_id" in row for row in results)

def test_ranking_contains_evidence():
    candidates = load_resumes()
    req = extract_requirements("React developers with 3+ years experience")
    ranked = rank_candidates(candidates, req, limit=10)
    assert len(ranked) == 10
    assert "score" in ranked[0]
    assert "evidence" in ranked[0]
    assert "gaps" in ranked[0]

def test_compare_and_interview_questions():
    candidates = load_resumes()
    req = extract_requirements("React developers with 3+ years experience")
    ranked = rank_candidates(candidates, req, limit=10)
    comparison = compare_candidates([ranked[0]["candidate_id"]], ranked)
    assert len(comparison) == 1
    questions = generate_interview_questions(ranked[0], req)
    assert len(questions) >= 1

def test_interview_questions_use_candidate_evidence():
    candidates = load_resumes()
    req = extract_requirements(
        "Find React developers with 3+ years experience"
    )

    alex = next(c for c in candidates if c["candidate_id"] == "C001")
    questions = generate_interview_questions(alex, req)

    assert questions
    assert any(
        "customer portal" in q.lower()
        or "28%" in q
        or "12,000" in q
        for q in questions
    )

def test_interview_questions_use_proper_skill_names():
    candidates = load_resumes()
    requirements = extract_requirements(
        "Find React developers with 3+ years experience. "
        "TypeScript is mandatory."
    )

    candidate = next(
        c for c in candidates
        if c["candidate_id"] == "C001"
    )
    questions = generate_interview_questions(candidate, requirements)
    combined = " ".join(questions)

    assert "React" in combined
    assert "TypeScript" in combined
    assert "using react" not in combined
    assert "using typescript" not in combined


def test_interview_questions_include_technical_depth():
    candidates = load_resumes()
    requirements = extract_requirements(
        "Find React developers with 3+ years experience. "
        "TypeScript is mandatory."
    )

    candidate = next(
        c for c in candidates
        if c["candidate_id"] == "C001"
    )
    questions = generate_interview_questions(candidate, requirements)
    combined = " ".join(questions).lower()

    assert any(
        term in combined
        for term in ("profil", "optimiz", "slow react")
    )
    assert any(
        term in combined
        for term in ("union type", "generics", "type design", "invalid states")
    )