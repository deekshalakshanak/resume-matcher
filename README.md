<<<<<<< HEAD
# Resume–Job Description Matcher

NLP-powered matching system using **Sentence Transformers** + **Scikit-learn**.

---

## Setup in VS Code

### 1. Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the `all-MiniLM-L6-v2` model (~80 MB). Cached after that.

---

## Run the demo
```bash
python matcher.py
```
Sample resume and JD are built-in. You'll see a scored report in the terminal.

---

## CLI usage

**Single match:**
```bash
python cli.py --resume my_resume.txt --jd job_description.txt
```

**Rank against multiple JDs:**
```bash
python cli.py --resume my_resume.txt --jd job1.txt job2.txt job3.txt
```

**Match against an entire folder of JDs:**
```bash
python cli.py --resume my_resume.txt --jd_dir ./jobs/
```

**Save results to JSON:**
```bash
python cli.py --resume my_resume.txt --jd_dir ./jobs/ --output results.json
```

**Show only top 3 matches:**
```bash
python cli.py --resume my_resume.txt --jd_dir ./jobs/ --top 3
```

---

## How it works

| Component | What it does |
|---|---|
| **Text Preprocessing** | Lowercasing, punctuation removal, whitespace normalization |
| **Section Extraction** | Regex-based parsing of Skills / Experience / Education sections |
| **Sentence Transformers** | `all-MiniLM-L6-v2` generates 384-dim semantic embeddings |
| **Cosine Similarity** | Measures angle between resume and JD embedding vectors |
| **TF-IDF Vectorizer** | Bag-of-ngrams fallback similarity (scikit-learn) |
| **Keyword Overlap** | Set intersection of JD keywords found / missing in resume |
| **Composite Score** | Weighted blend: 60% semantic + 20% TF-IDF + 20% keyword |

---

## Project Structure

```
resume_matcher/
├── matcher.py        # Core NLP pipeline (preprocessing, embeddings, scoring)
├── cli.py            # Command-line interface for file-based matching
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Customising weights

```python
from matcher import ResumeMatcher

matcher = ResumeMatcher()
result = matcher.score(
    resume_text,
    job_desc_text,
    weights={"semantic": 0.7, "tfidf": 0.1, "keyword": 0.2}
)
```

## Swapping the model

Any Sentence Transformers model works:
```python
matcher = ResumeMatcher(model_name="all-mpnet-base-v2")   # larger, more accurate
matcher = ResumeMatcher(model_name="paraphrase-MiniLM-L3-v2")  # faster, smaller
```
Browse models: https://www.sbert.net/docs/pretrained_models.html
=======
# resume-matcher
>>>>>>> 7a3f40b7e5bdddd19919adf7e75a8cfcb6f65845
