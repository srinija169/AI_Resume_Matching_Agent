from requirements import extract_requirements, merge_requirements


from requirements import extract_requirements, merge_requirements


def test_vague_frontend_role_uses_transparent_defaults():
    reqs = extract_requirements("A frontend developer job description")

    assert set(reqs["must_have"]["skills"]) == {
        "react", "javascript", "html/css"
    }
    assert set(reqs["nice_to_have"]["skills"]) == {
        "typescript", "next.js", "git"
    }
    assert reqs["must_have"]["minimum_experience_years"] == 0
    assert reqs["detected_role"] == "frontend"
    assert reqs["assumptions"]


def test_explicit_mandatory_typescript_is_preserved():
    reqs = extract_requirements(
        "Frontend developer. TypeScript is a must-have."
    )

    assert "typescript" in reqs["must_have"]["skills"]
    assert "typescript" not in reqs["nice_to_have"]["skills"]


def test_refinement_promotes_typescript_to_must_have():
    current = extract_requirements("A frontend developer job description")
    update = extract_requirements(
        "Now require TypeScript as a must-have"
    )

    merged = merge_requirements(current, update)

    assert "typescript" in merged["must_have"]["skills"]
    assert "typescript" not in merged["nice_to_have"]["skills"]


def test_explicit_experience_minimum_is_extracted():
    reqs = extract_requirements(
        "Frontend developer with at least 3 years of experience"
    )

    assert reqs["must_have"]["minimum_experience_years"] == 3
