
"""Deterministic job-description requirement extraction."""
import re
from typing import Any


SKILL_ALIASES = {
    "react": ["react", "react.js", "reactjs"],
    "typescript": ["typescript", "ts"],
    "javascript": ["javascript", "js"],
    "python": ["python"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "node.js": ["node.js", "nodejs", "node"],
    "sql": ["sql", "postgresql", "mysql", "sqlite"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel"],
    "pandas": ["pandas"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "langchain": ["langchain"],
    "langgraph": ["langgraph"],
    "machine learning": ["machine learning"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "next.js": ["next.js", "nextjs"],
    "html/css": ["html/css", "html", "css"],
    "git": ["git", "github"],
    "java": ["java"],
    "c++": ["c++"],
    "communication": ["communication", "stakeholder communication"],
    "leadership": ["leadership", "team lead", "mentoring"],
}


ROLE_PROFILES = {
    "frontend": {
        "must_have": ["react", "javascript", "html/css"],
        "nice_to_have": ["typescript", "next.js", "git"],
    },
    "data analyst": {
        "must_have": ["sql", "python", "power bi"],
        "nice_to_have": ["excel", "pandas", "tableau"],
    },
    "backend": {
        "must_have": ["python", "fastapi"],
        "nice_to_have": ["sql", "docker", "aws"],
    },
}


def _contains(text: str, term: str) -> bool:
    """Match a skill alias without matching it inside a larger word."""
    return bool(
        re.search(
            r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#.])",
            text,
            re.I,
        )
    )



def _detect_role(text: str) -> str | None:
    patterns = {
        "frontend": (
            r"\bfront[\s-]?end\b",
            r"\bui developer\b",
            r"\buser interface developer\b",
            r"\breact developer\b",
        ),
        "data analyst": (
            r"\bdata analyst(?:s)?\b",
            r"\bdata analytics\b",
            r"\bbusiness analyst\b",
            r"\bdata analysis\b",
        ),
        "backend": (
            r"\bback[\s-]?end\b",
            r"\bserver[- ]side developer\b",
            r"\bapi developer\b",
        ),
    }

    for role, role_patterns in patterns.items():
        if any(re.search(pattern, text, re.I) for pattern in role_patterns):
            return role

    return None


def _extract_years(text: str) -> int:
    patterns = (
        r"(\d+)\s*\+\s*(?:years?|yrs?)",
        r"(?:at least|minimum of|min(?:imum)?)\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:or more)\s*(?:years?|yrs?)",
    )
    values = []
    for pattern in patterns:
        values.extend(int(x) for x in re.findall(pattern, text, re.I))
    return max(values, default=0)


def _is_explicitly_mandatory(skill: str, text: str) -> bool:
    aliases = SKILL_ALIASES.get(skill, [skill])
    mandatory = r"(?:mandatory|required|must-have|must have)"

    for alias in aliases:
        escaped = re.escape(alias)
        if re.search(rf"{escaped}.{{0,50}}{mandatory}", text, re.I):
            return True
        if re.search(rf"{mandatory}.{{0,50}}{escaped}", text, re.I):
            return True

    return False


def extract_requirements(jd: str) -> dict[str, Any]:
    """Extract role defaults, explicitly requested skills, and experience."""
    text = jd.lower()

    explicit_skills = [
        canonical
        for canonical, aliases in SKILL_ALIASES.items()
        if any(_contains(text, alias) for alias in aliases)
    ]

    role = _detect_role(text)
    min_years = _extract_years(text)

    explicit_mandatory = {
        skill for skill in explicit_skills
        if _is_explicitly_mandatory(skill, text)
    }

    profile = ROLE_PROFILES.get(role, {})
    default_must = set(profile.get("must_have", []))
    default_nice = set(profile.get("nice_to_have", []))

    # Explicitly requested skills become required unless clearly optional.
    optional_words = r"\b(?:nice.to.have|optional|bonus|preferred)\b"
    explicitly_optional = set()

    for skill in explicit_skills:
        aliases = SKILL_ALIASES.get(skill, [skill])
        for alias in aliases:
            escaped = re.escape(alias)
            if re.search(
                rf"{escaped}.{{0,40}}{optional_words}|"
                rf"{optional_words}.{{0,40}}{escaped}",
                text,
                re.I,
            ):
                explicitly_optional.add(skill)
                break

    if role:
        must_skills = default_must | (
            set(explicit_skills) - default_nice - explicitly_optional
        )
        nice_skills = (
            default_nice | explicitly_optional
            | (set(explicit_skills) & default_nice)
        ) - must_skills
    else:
        # Generic searches require all explicitly requested skills.
        must_skills = (
            set(explicit_skills) - explicitly_optional
        ) | explicit_mandatory
        nice_skills = explicitly_optional - must_skills

    # Explicit mandatory wording always takes priority.
    must_skills |= explicit_mandatory
    nice_skills -= must_skills
    all_skills = must_skills | nice_skills

    assumptions = []
    if role == "frontend":
        assumptions.append(
            "Frontend defaults: React, JavaScript, and HTML/CSS are required; "
            "TypeScript, Next.js, and Git are preferred."
        )
    elif role == "data analyst":
        assumptions.append(
            "Data analyst defaults: SQL, Python, and Power BI are required; "
            "Excel, Pandas, and Tableau are preferred."
        )
    elif role == "backend":
        assumptions.append(
            "Backend defaults: Python and FastAPI are required; "
            "SQL, Docker, and AWS are preferred."
        )

    if min_years == 0:
        assumptions.append(
            "No minimum experience was specified, so the minimum remains 0 years."
        )

    return {
        "must_have": {
            "skills": sorted(must_skills),
            "minimum_experience_years": min_years,
        },
        "nice_to_have": {"skills": sorted(nice_skills)},
        "all_requested_skills": sorted(all_skills),
        "source_text": jd,
        "detected_role": role,
        "assumptions": assumptions,
    }


def merge_requirements(
    current: dict[str, Any] | None,
    update: dict[str, Any],
) -> dict[str, Any]:
    """Merge conversational refinements into existing requirements."""
    if not current:
        return update

    result = dict(current)
    must = dict(current.get("must_have", {}))
    nice = dict(current.get("nice_to_have", {}))
    update_text = update.get("source_text", "")

    must_skills = set(must.get("skills", []))
    nice_skills = set(nice.get("skills", []))

    for skill in update.get("all_requested_skills", []):
        if _is_explicitly_mandatory(skill, update_text):
            must_skills.add(skill)
            nice_skills.discard(skill)
        elif skill not in must_skills and skill not in nice_skills:
            nice_skills.add(skill)

    new_min = update.get("must_have", {}).get(
        "minimum_experience_years", 0
    )
    if new_min:
        must["minimum_experience_years"] = max(
            int(must.get("minimum_experience_years", 0) or 0),
            int(new_min),
        )

    must["skills"] = sorted(must_skills)
    nice["skills"] = sorted(nice_skills - must_skills)

    result["must_have"] = must
    result["nice_to_have"] = nice
    result["all_requested_skills"] = sorted(
        must_skills | nice_skills
    )
    result["source_text"] = (
        current.get("source_text", "") + "\n" + update_text
    ).strip()
    result["detected_role"] = (
        update.get("detected_role") or current.get("detected_role")
    )
    result["assumptions"] = list(dict.fromkeys(
        current.get("assumptions", []) + update.get("assumptions", [])
    ))

    return result
