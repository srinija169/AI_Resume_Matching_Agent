# AI Resume Matching Agent

A conversational AI resume-matching agent built with Python and LangGraph. This educational project demonstrates job requirement extraction, local resume retrieval, evidence-aware candidate ranking, candidate comparison, ranking explanations, iterative requirement refinement, and deeper candidate screening with interview questions.

The project uses **100 synthetic resume profiles**, a local TF-IDF retrieval layer, and configurable ranking rules. The local workflow does not require an external API key or vector database.

## 1. Project Overview

The AI Resume Matching Agent demonstrates how a recruiter can search and evaluate candidate profiles against job-related requirements through a conversational workflow.

### Key features

- Search a dataset of 100 synthetic resumes.
- Extract required skills, preferred skills, and minimum experience from job descriptions.
- Retrieve relevant resumes using local TF-IDF text retrieval.
- Rank candidates using configurable scoring rules.
- Identify matching skills, strengths, gaps, and supporting resume evidence.
- Compare the top three candidates.
- Explain ranking differences using candidate evidence and configured scoring rules.
- Refine requirements through natural-language feedback and recalculate rankings.
- Deep-screen a shortlist of up to 10 candidates.
- Generate candidate-specific interview questions.
- Produce provisional screening recommendations.
- Maintain conversational workflow state across turns while the application process is running.
- Support human review rather than making automatic final hiring decisions.

## 2. Assignment Requirements and Implementation Status

| Requirement | Implementation status |
|---|---|
| LangGraph agent architecture | Implemented |
| Job requirement extraction | Implemented |
| Resume retrieval | Implemented using local TF-IDF |
| Initial screening of 100 resumes | Supported by the 100-profile dataset |
| Candidate ranking | Implemented |
| Deep analysis of the top 10 | Implemented |
| Candidate comparison | Implemented |
| Ranking explanations | Implemented |
| Conversational requirement refinement | Implemented |
| Interview question generation | Implemented |
| Provisional screening recommendation | Implemented |
| Final human hiring decision | Reserved for human review |
| Automated tests | 17 tests passing in the current development environment |

The agent returns up to 10 candidates for the initial shortlist from the available resume corpus. Deep screening evaluates the current shortlist and generates additional evidence-based analysis.

## 3. Technology Stack

- **Python:** Core implementation.
- **LangGraph:** Workflow graph and conversational state management.
- **LangChain message types:** Conversation messages.
- **scikit-learn:** TF-IDF vectorization and cosine similarity for resume retrieval.
- **Streamlit:** User interface.
- **pytest:** Automated testing.

The local implementation uses deterministic requirement extraction and scoring rules rather than requiring a hosted language model. No external API key or vector database is required for the local demo.

## 4. How the Agent Works

The main workflow follows this sequence:

1. **Parse Request:** Identify whether the user wants to search, compare candidates, explain a ranking, refine requirements, generate interview questions, or perform deep screening.
2. **Extract Requirements:** Identify job-related must-have skills, nice-to-have skills, and minimum experience.
3. **Search Resumes:** Retrieve relevant candidate profiles from the local resume dataset using TF-IDF.
4. **Rank Candidates:** Calculate scores using the configured criteria and evaluate must-have requirements.
5. **Generate Report:** Present rankings, strengths, gaps, supporting evidence, and relevant explanations.
6. **Human Feedback:** Allow the recruiter to refine criteria or request additional analysis in a subsequent conversational turn.

The LangGraph state and in-memory checkpointer maintain relevant context across turns while the process is running. Conversation state is not designed to survive process restarts.

The feedback mechanism supports iterative conversation; it is not a blocking human-approval interrupt.

## 5. Architecture Diagram

The primary workflow is:

```text
             START
               |
               v
         Parse Request
               |
               v
      Extract Requirements
               |
               v
         Search Resumes
            (TF-IDF)
               |
               v
       Rank Candidates
               |
               v
        Generate Report
               |
               v
      Human Feedback Loop
               |
               v
              END
```

Candidate comparison, ranking explanations, requirement refinement, and interview-question generation are supported through the agent's conversational workflow.

The graph preserves relevant state across conversational turns, enabling follow-up requests to use the active requirements and candidate results.

## 6. Candidate Ranking and Explainability

Candidate scores are calculated using the following configured weights:

| Criterion | Weight |
|---|---:|
| Must-have skill coverage | 60% |
| Experience | 15% |
| Nice-to-have skill coverage | 15% |
| Project evidence coverage | 10% |
| **Total** | **100%** |

The scoring formula is:

```text
score = 100 * (
    0.60 * must_coverage
  + 0.15 * experience_score
  + 0.15 * nice_coverage
  + 0.10 * project_coverage
)
```

The experience component is normalized using the configured minimum experience and five-year scale, subject to the implementation's handling of minimum-experience requirements. Must-have qualification checks are also used to identify candidates who do not satisfy all configured mandatory criteria.

### Explainability features

The agent provides:

- Overall candidate scores.
- Must-have qualification checks.
- Matching skills.
- Strengths and gaps.
- Supporting resume evidence.
- Explanations of ranking differences.
- Updated results after requirement refinement.
- Interview questions to validate relevant skills and experience.

The scoring weights are illustrative and have not been validated as a real-world hiring assessment.

A skill that is not explicitly documented in a resume is treated as **not evidenced in the supplied resume**, rather than proof that the candidate lacks that skill.

## 7. Initial Screening and Deep Screening

### Initial screening

The initial screening stage retrieves and ranks candidates against the job requirements and returns up to 10 candidates from the 100-profile synthetic resume dataset.

For example, a data analyst search may use SQL, Python, and Power BI as mandatory skills, with Excel, Pandas, and Tableau as preferred skills.

The agent reports candidate scores, must-have qualification status, matching skills, evidence, and gaps.

### Deep screening

The deep-screening stage analyzes the current shortlist of up to 10 candidates and produces:

- Candidate ranking details.
- Strengths, gaps, and supporting evidence.
- Must-have qualification status.
- Provisional recommendations for further consideration.
- Candidate-by-candidate screening statuses.
- Suggested interview questions.
- A reminder that final decisions require human review.

The agent can also deep-screen a specific candidate independently when requested.

Deep screening does not independently verify candidate claims, credentials, employment history, or skill proficiency. Its recommendations are based on the supplied synthetic resume data and configured rules.

## 8. Example Conversations

The following prompts demonstrate supported workflows in the application or agent interface.

### Initial search

```text
Find and rank the top 10 candidates for a data analyst role requiring SQL, Python, and Power BI.
```

### Search with experience requirements

```text
Find React developers with 3+ years experience.
```

### Compare candidates

```text
Compare the top 3 candidates.
```

### Explain a ranking

```text
Explain why the top candidate ranked higher.
```

### Refine requirements

```text
Now require TypeScript as a must-have.
```

### Deep-screen the shortlist

```text
Deep-screen all 10 candidates with strengths, gaps, and interview questions.
```

### Deep-screen an individual candidate

```text
Deep-screen C083 alone and verify that Power BI is flagged as missing.
```

### Generate interview questions

```text
Generate interview questions for candidate C077.
```

The current conversational intent router uses deterministic keyword patterns. Some alternative phrasings may not be recognized correctly.

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later, subject to dependency compatibility.
- pip.
- Windows PowerShell or another terminal.
- No external API key required for the local demo.

### Step 1: Open the project folder

```powershell
cd "C:\Users\HP\Desktop\AIRTRIBE\RAG project\matching__agent"
```

### Step 2: Activate the virtual environment

If `.venv` already exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

If it does not exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Streamlit interface

If `app.py` is present in the project root:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal if the browser does not open automatically.

### Step 5: Run the automated tests

```powershell
python -m pytest -q
```

The current verified development run completed with:

```text
17 passed in 2.74s
```

This is the observed result from the development environment; results may vary if the code, dependencies, or environment change.

## 10. Project Structure

The main project files are:

```text
matching__agent/
├── app.py
├── matching_agent.py
├── requirements.py
├── rag_search.py
├── ranking.py
├── tools.py
├── config.py
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── data/
│   └── resumes.json
├── diagrams/
├── tests/
│   ├── test_agent.py
│   ├── test_conversations.py
│   └── test_requirements.py
└── README.md
```

The `data/resumes.json` file contains 100 synthetic resume profiles. The `diagrams/` directory contains project diagram assets, if present.

## 11. Testing

The current automated test suite contains **17 passing tests**.

The tests cover selected behaviors, including:

- Requirement extraction.
- Requirement refinement and merging.
- Candidate ranking and mandatory-skill handling.
- Conversational interactions.
- Candidate comparison.
- Ranking explanations.
- Deep-screening behavior.
- Interview-question generation.

Run the tests from the project root:

```powershell
python -m pytest -q
```

A passing test suite verifies the tested behaviors, not every possible conversation or dataset.

## 12. Important Limitations

This project is an educational prototype, not a production recruitment system.

1. **Synthetic data:** The 100 candidate profiles are synthetic and should not be interpreted as real applicants.
2. **Lexical retrieval:** TF-IDF uses textual similarity rather than semantic embedding-based retrieval.
3. **Limited requirement extraction:** Requirements are extracted using deterministic patterns and a limited skill vocabulary. Unusual phrasing or context may be missed.
4. **Illustrative scoring:** Scoring weights and ranking rules are examples and have not been validated as hiring standards.
5. **Rule-based intent routing