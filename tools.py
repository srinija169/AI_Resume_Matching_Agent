"""Agent tool adapters."""
from langchain_core.tools import tool
from rag_search import ResumeRetriever, load_resumes
from requirements import extract_requirements
from ranking import compare_candidates as compare_impl, generate_interview_questions as questions_impl

_RETRIEVER = ResumeRetriever()
_RESUMES = load_resumes()

@tool
def search_resumes(query: str, limit: int = 100) -> list[dict]:
    """Search the local synthetic resume corpus using TF-IDF retrieval."""
    return _RETRIEVER.search(query, limit)

@tool
def extract_requirements_tool(jd: str) -> dict:
    """Extract must-have and nice-to-have skills and experience from a job description."""
    return extract_requirements(jd)

def compare_candidates(candidate_ids: list[str], shortlist: list[dict]) -> list[dict]:
    """Compare specified candidate IDs from the current shortlist."""
    return compare_impl(candidate_ids, shortlist)

def generate_interview_questions(candidate_id: str, shortlist: list[dict], requirements: dict) -> list[str]:
    """Generate role-specific screening questions for a shortlisted candidate."""
    candidate = next((c for c in shortlist if c["candidate_id"] == candidate_id), None)
    if candidate is None:
        return [f"Candidate {candidate_id} is not in the current shortlist."]
    return questions_impl(candidate, requirements)
