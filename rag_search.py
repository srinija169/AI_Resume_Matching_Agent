"""Small local RAG-like retriever for synthetic resumes.

Uses TF-IDF lexical retrieval so the demo runs without API keys or vector DB.
For production, replace with an embedding-based vector store and add access
controls, document provenance, and evaluation.
"""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "resumes.json"

def load_resumes(path: Path = DATA_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

class ResumeRetriever:
    def __init__(self, resumes: list[dict] | None = None):
        self.resumes = resumes if resumes is not None else load_resumes()
        self.texts = [
            " ".join([
                r.get("resume_text", ""),
                r.get("summary", ""),
                " ".join(r.get("skills", [])),
                " ".join(p.get("details", "") for p in r.get("projects", [])),
            ])
            for r in self.resumes
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def search(self, query: str, limit: int = 100) -> list[dict]:
        if not self.resumes or not query.strip():
            return self.resumes[:limit]
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [
            {**self.resumes[i], "retrieval_score": round(float(score), 4)}
            for i, score in ranked[:limit]
        ]
