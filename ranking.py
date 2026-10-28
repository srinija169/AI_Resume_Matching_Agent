"""Evidence-aware candidate scoring and screening utilities."""
from typing import Any

def _skill_set(candidate: dict) -> set[str]:
    return {str(s).strip().lower() for s in candidate.get("skills", [])}

def score_candidate(candidate: dict, requirements: dict[str, Any]) -> dict[str, Any]:
    must = requirements.get("must_have", {})
    nice = requirements.get("nice_to_have", {})
    candidate_skills = _skill_set(candidate)
    must_skills = {s.lower() for s in must.get("skills", [])}
    nice_skills = {s.lower() for s in nice.get("skills", [])}
    years = int(candidate.get("experience_years", 0) or 0)
    min_years = int(must.get("minimum_experience_years", 0) or 0)

    must_coverage = len(must_skills & candidate_skills) / max(1, len(must_skills)) if must_skills else 1.0
    # Meeting the minimum earns a baseline, while additional
    # relevant experience can contribute a smaller bonus.
    if min_years:
        experience_score = (
            0.7 + 0.3 * min(1.0, years / (min_years + 3))
            if years >= min_years
            else 0.7 * years / min_years
        )
    else:
        experience_score = min(1.0, years / 5)
    nice_coverage = len(nice_skills & candidate_skills) / max(1, len(nice_skills)) if nice_skills else 0.0
    project_text = " ".join(p.get("details", "") for p in candidate.get("projects", [])).lower()
    project_coverage = sum(1 for skill in (must_skills | nice_skills) if skill in project_text) / max(1, len(must_skills | nice_skills))

    score = 100 * (
        0.60 * must_coverage +
        0.15 * experience_score +
        0.15 * nice_coverage +
        0.10 * project_coverage
    )

    missing = sorted(must_skills - candidate_skills)
    missing_display = [_display_skill(skill) for skill in missing]
    experience_gap = max(0, min_years - years)
    gaps = []
    if missing:
        gaps.append("Required skills not evidenced in the supplied resume: " + ", ".join(missing_display))
    if experience_gap:
        gaps.append(f"Listed experience is {experience_gap} year(s) below the requested minimum")
    if not gaps:
        gaps.append("No explicit must-have gap found in the supplied resume data")
    strengths = []
    matched = sorted(candidate_skills & (must_skills | nice_skills))
    if matched:
        strengths.append("Listed matching skills: " + ", ".join(matched))
    if years >= min_years:
        strengths.append(f"{years} years listed experience meets the {min_years}-year minimum")
    evidence = [
        f"Resume lists {years} years of experience",
        "Skills listed: " + (", ".join(sorted(candidate.get("skills", []))) or "none"),
    ]
    evidence.extend(p.get("details", "") for p in candidate.get("projects", []))
    return {
        **candidate,
        "score": round(score, 1),
        "must_have_pass": not missing and experience_gap == 0,
        "strengths": strengths,
        "gaps": gaps,
        "evidence": evidence,
        "matched_skills": matched,
        "missing_required_skills": missing,
    }

def rank_candidates(candidates: list[dict], requirements: dict, limit: int | None = None) -> list[dict]:
    ranked = [score_candidate(c, requirements) for c in candidates]
    ranked.sort(key=lambda c: (c["must_have_pass"], c["score"]), reverse=True)
    return ranked[:limit] if limit is not None else ranked

def compare_candidates(candidate_ids: list[str], shortlist: list[dict]) -> list[dict]:
    wanted = set(candidate_ids)
    return [
        {
            "candidate_id": c["candidate_id"],
            "name": c["name"],
            "score": c["score"],
            "experience_years": c["experience_years"],
            "matched_skills": c["matched_skills"],
            "gaps": c["gaps"],
        }
        for c in shortlist if c["candidate_id"] in wanted
    ]

def _display_skill(skill: str) -> str:
    """Format common technical skill names for readable interview questions."""
    names = {
        "react": "React",
        "typescript": "TypeScript",
        "javascript": "JavaScript",
        "next.js": "Next.js",
        "node.js": "Node.js",
        "python": "Python",
        "sql": "SQL",
        "aws": "AWS",
        "html/css": "HTML/CSS",
        "docker": "Docker",
    }
    return names.get(skill.strip().lower(), skill.strip())

def generate_interview_questions(
    candidate: dict,
    requirements: dict[str, Any],
) -> list[str]:
    """Generate candidate-specific questions grounded in supplied resume evidence."""
    must = requirements.get("must_have", {})
    nice = requirements.get("nice_to_have", {})

    required_skills = must.get("skills", [])
    nice_skills = nice.get("skills", [])
    candidate_skills = _skill_set(candidate)

    missing = candidate.get(
        "missing_required_skills",
        sorted({s.lower() for s in required_skills} - candidate_skills),
    )

    projects = candidate.get("projects", [])
    experience = int(candidate.get("experience_years", 0) or 0)
    minimum_years = int(must.get("minimum_experience_years", 0) or 0)

    questions = []

    # Start with actual projects rather than a generic project question.
    for project in projects[:2]:
        name = project.get("name", "this project")
        details = project.get("details", "").strip()

        if details:
            questions.append(
                f"For your {name} project, your resume states: "
                f"'{details}' What was your individual contribution, "
                "and how did you measure the outcome?"
            )
        else:
            questions.append(
                f"Walk me through your {name} project and explain "
                "your individual contribution."
            )

    # Probe must-have skills that are listed on the resume.
    for skill in required_skills:
        if skill.lower() in candidate_skills:
            display_skill = _display_skill(skill)
            questions.append(
                f"Describe a technical decision you made using "
                f"{display_skill}, the alternatives you considered, "
                "and how you tested the result."
            )

    # Add deeper questions based on the candidate's evidenced skills.
    if "react" in candidate_skills or any(s.lower() == "react" for s in required_skills + nice_skills):
        questions.append(
            "How would you investigate a slow React page? "
            "Explain which profiling tools and measurements you would use "
            "before deciding what to optimize."
        )
        questions.append(
            "How do you test React components and user interactions? "
            "Describe a test you would write and what failures it should catch."
        )

    if any(s.lower() == "typescript" for s in required_skills):
        questions.append(
            "How would you design TypeScript types for an API response "
            "that can represent both success and error states? "
            "How would you make invalid states harder to represent?"
        )
        questions.append(
            "Explain when you would use a union type, an interface, "
            "and generics in a TypeScript application. Give a practical example."
        )

    questions.append(
        "Choose one result or improvement mentioned in your resume. "
        "How was it measured, what was the baseline, and what evidence "
        "shows that your work caused the improvement?"
    )

    # Check required skills that the supplied resume does not list.
    for skill in missing:
        display_skill = _display_skill(skill)
        questions.append(
            f"The supplied resume does not explicitly document "
            f"{display_skill}. Can you describe any hands-on experience "
            f"you have with it, including a project or practical example?"
        )

    # Test experience requirements without assuming that a resume gap
    # proves the candidate lacks the experience.
    if minimum_years and experience < minimum_years:
        questions.append(
            f"The resume lists {experience} years of experience, while "
            f"the role requests at least {minimum_years}. Can you clarify "
            "your relevant experience and the dates of that work?"
        )

    # Use nice-to-have skills only when they are actually listed.
    for skill in nice_skills:
        display_skill = _display_skill(skill)
        if skill.lower() in candidate_skills:
            questions.append(
                f"How have you used {display_skill} in a real project, and what "
                "trade-offs or limitations did you encounter?"
            )

    # Avoid duplicate questions while keeping the original order.
    unique_questions = list(dict.fromkeys(questions))

    # Always return a useful set, even when resume details are sparse.
    if not unique_questions:
        unique_questions = [
            "Which experience from your resume is most relevant to this "
            "role, and what evidence demonstrates your contribution?",
            "How do you test your work and investigate defects?",
            "What technical skill relevant to this role would you most "
            "like to demonstrate during an interview?",
        ]

    return unique_questions[:10]
    