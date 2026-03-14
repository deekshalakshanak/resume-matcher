"""
Resume-Job Description Matcher
Using Sentence Transformers + Scikit-learn NLP Pipeline
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalize input text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)          # remove punctuation
    text = re.sub(r'\d+', ' ', text)               # remove standalone numbers
    text = re.sub(r'\s+', ' ', text).strip()       # collapse whitespace
    return text


def extract_sections(resume_text: str) -> dict:
    """Loosely extract key resume sections by keyword headers."""
    sections = {"skills": "", "experience": "", "education": "", "full": resume_text}

    skill_match = re.search(
        r'(skills?|technical skills?|core competencies)(.*?)(experience|education|projects|$)',
        resume_text, re.IGNORECASE | re.DOTALL
    )
    exp_match = re.search(
        r'(experience|work history|employment)(.*?)(education|skills?|projects|$)',
        resume_text, re.IGNORECASE | re.DOTALL
    )
    edu_match = re.search(
        r'(education|academic background)(.*?)(experience|skills?|projects|certifications|$)',
        resume_text, re.IGNORECASE | re.DOTALL
    )

    if skill_match:
        sections["skills"] = skill_match.group(2).strip()
    if exp_match:
        sections["experience"] = exp_match.group(2).strip()
    if edu_match:
        sections["education"] = edu_match.group(2).strip()

    return sections


# ─────────────────────────────────────────────
#  EMBEDDING + SIMILARITY
# ─────────────────────────────────────────────

class ResumeMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[INFO] Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        print("[INFO] Model loaded successfully.\n")

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def semantic_similarity(self, resume: str, job_desc: str) -> float:
        embeddings = self.get_embeddings([resume, job_desc])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(score)

    def tfidf_similarity(self, resume: str, job_desc: str) -> float:
        tfidf_matrix = self.tfidf.fit_transform([resume, job_desc])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return float(score)

    def keyword_overlap(self, resume: str, job_desc: str) -> dict:
        """Find matching and missing keywords between resume and JD."""
        resume_tokens = set(preprocess_text(resume).split())
        jd_tokens = set(preprocess_text(job_desc).split())

        # Filter short/stopwords
        stop = {"and", "the", "to", "of", "in", "a", "an", "for", "with",
                "is", "are", "be", "will", "on", "at", "by", "or", "this"}
        jd_keywords = {w for w in jd_tokens if len(w) > 3 and w not in stop}

        matched = jd_keywords & resume_tokens
        missing = jd_keywords - resume_tokens

        overlap_pct = len(matched) / len(jd_keywords) * 100 if jd_keywords else 0
        return {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "overlap_percent": round(overlap_pct, 2)
        }

    def score(self, resume_text: str, job_desc_text: str,
              weights: dict = None) -> dict:
        """
        Full scoring pipeline.
        weights: dict with keys 'semantic', 'tfidf', 'keyword'
                 (must sum to 1.0, default: 0.6 / 0.2 / 0.2)
        """
        if weights is None:
            weights = {"semantic": 0.6, "tfidf": 0.2, "keyword": 0.2}

        # Preprocess
        clean_resume = preprocess_text(resume_text)
        clean_jd = preprocess_text(job_desc_text)

        # Scores
        sem_score = self.semantic_similarity(clean_resume, clean_jd)
        tfidf_score = self.tfidf_similarity(clean_resume, clean_jd)
        kw_data = self.keyword_overlap(resume_text, job_desc_text)
        kw_score = kw_data["overlap_percent"] / 100

        # Weighted composite
        composite = (
            weights["semantic"] * sem_score +
            weights["tfidf"]   * tfidf_score +
            weights["keyword"] * kw_score
        )

        # Section-level breakdown
        sections = extract_sections(resume_text)
        section_scores = {}
        for name, content in sections.items():
            if name == "full" or not content.strip():
                continue
            s = self.semantic_similarity(preprocess_text(content), clean_jd)
            section_scores[name] = round(s * 100, 2)

        return {
            "composite_score": round(composite * 100, 2),
            "semantic_score":  round(sem_score * 100, 2),
            "tfidf_score":     round(tfidf_score * 100, 2),
            "keyword_overlap": kw_data,
            "section_scores":  section_scores,
        }


# ─────────────────────────────────────────────
#  PRETTY PRINT
# ─────────────────────────────────────────────

def print_report(result: dict):
    bar = lambda v, w=40: "█" * int(v / 100 * w) + "░" * (w - int(v / 100 * w))

    print("\n" + "=" * 55)
    print("          RESUME ↔ JOB MATCH REPORT")
    print("=" * 55)

    cs = result["composite_score"]
    label = ("🔴 Low Match" if cs < 40 else
             "🟡 Moderate Match" if cs < 65 else
             "🟢 Strong Match")

    print(f"\n  COMPOSITE SCORE : {cs:.1f}%  {label}")
    print(f"  {bar(cs)}")

    print(f"\n  Semantic (ST)   : {result['semantic_score']:.1f}%  {bar(result['semantic_score'])}")
    print(f"  TF-IDF          : {result['tfidf_score']:.1f}%  {bar(result['tfidf_score'])}")
    kw = result["keyword_overlap"]
    print(f"  Keyword Overlap : {kw['overlap_percent']:.1f}%  {bar(kw['overlap_percent'])}")

    if result["section_scores"]:
        print("\n  ── Section Breakdown ──")
        for sec, sc in result["section_scores"].items():
            print(f"  {sec.capitalize():12s}: {sc:.1f}%  {bar(sc, 20)}")

    print(f"\n  ── Matched Keywords ({len(kw['matched'])}) ──")
    print("  " + ", ".join(kw["matched"][:20]) + ("..." if len(kw["matched"]) > 20 else ""))

    print(f"\n  ── Missing Keywords ({len(kw['missing'])}) ──")
    print("  " + ", ".join(list(kw["missing"])[:20]) + ("..." if len(kw["missing"]) > 20 else ""))

    print("\n" + "=" * 55 + "\n")


# ─────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────

SAMPLE_RESUME = """
John Doe | john@email.com | linkedin.com/in/johndoe

SKILLS
Python, Machine Learning, Deep Learning, NLP, Scikit-learn,
TensorFlow, PyTorch, Pandas, NumPy, SQL, Git, Docker, REST APIs,
Data Visualization (Matplotlib, Seaborn), AWS (S3, EC2)

EXPERIENCE
Senior Data Scientist – ABC Corp (2021–Present)
- Built NLP pipelines for document classification using BERT and spaCy
- Developed recommendation engine improving CTR by 18%
- Deployed ML models via FastAPI on AWS ECS

Data Analyst – XYZ Ltd (2019–2021)
- Performed exploratory data analysis on customer churn datasets
- Created Tableau dashboards for business stakeholders
- Automated ETL workflows with Airflow

EDUCATION
B.Tech in Computer Science – IIT Madras, 2019
"""

SAMPLE_JD = """
We are looking for a Machine Learning Engineer to join our AI team.

Requirements:
- Strong proficiency in Python and ML frameworks (TensorFlow, PyTorch)
- Experience with NLP techniques: tokenization, embeddings, transformers
- Hands-on with Scikit-learn, model training, evaluation, and deployment
- Familiarity with REST APIs and containerization (Docker, Kubernetes)
- Experience with cloud platforms (AWS or GCP)
- Knowledge of MLOps practices and CI/CD pipelines
- Strong communication and collaboration skills

Nice to have:
- Experience with LLMs and prompt engineering
- Familiarity with vector databases (Pinecone, Weaviate)
"""

if __name__ == "__main__":
    matcher = ResumeMatcher()
    result = matcher.score(SAMPLE_RESUME, SAMPLE_JD)
    print_report(result)
