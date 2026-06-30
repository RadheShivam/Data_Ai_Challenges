import json
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Read Job Description
# -----------------------------
doc = Document("data/job_description.docx")

job_description = "\n".join(
    para.text.strip()
    for para in doc.paragraphs
    if para.text.strip()
)

# -----------------------------
# Read first candidate
# -----------------------------
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    candidate = json.loads(next(f))

profile = candidate["profile"]

career_text = " ".join(
    job["description"]
    for job in candidate["career_history"]
)

skills = ", ".join(
    skill["name"]
    for skill in candidate["skills"]
)

candidate_text = f"""
Headline:
{profile['headline']}

Summary:
{profile['summary']}

Career:
{career_text}

Skills:
{skills}
"""

# -----------------------------
# Generate embeddings
# -----------------------------
jd_embedding = model.encode(job_description)

candidate_embedding = model.encode(candidate_text)

# -----------------------------
# Semantic similarity
# -----------------------------
similarity = cosine_similarity(
    [jd_embedding],
    [candidate_embedding]
)[0][0]

print("=" * 80)
print("SEMANTIC MATCH")
print("=" * 80)

print("Candidate:", candidate["candidate_id"])
print(f"Similarity Score: {similarity:.4f}")