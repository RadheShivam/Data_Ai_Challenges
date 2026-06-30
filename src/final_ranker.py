import json
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import build_candidate_document

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading Job Description...")
doc = Document("data/job_description.docx")

job_description = "\n".join(
    para.text.strip()
    for para in doc.paragraphs
    if para.text.strip()
)

print("Generating Job Description embedding...")
job_embedding = model.encode(
    job_description,
    convert_to_numpy=True
)

print("Job embedding shape:", job_embedding.shape)
print("Setup completed successfully!")


# ------------------------------------------------
# Rule-Based Scoring
# ------------------------------------------------

required_skills = {
    "Python",
    "Milvus",
    "FAISS",
    "Pinecone",
    "Qdrant",
    "Weaviate",
    "OpenSearch",
    "Elasticsearch",
    "Fine-tuning LLMs",
    "LoRA",
}

career_keywords = {
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "embedding",
    "embeddings",
    "vector",
}


def rule_score(candidate):
    score = 0

    profile = candidate["profile"]
    skills = candidate["skills"]
    career = candidate["career_history"]
    signals = candidate["redrob_signals"]

    # Experience (15)
    exp = profile["years_of_experience"]

    if 5 <= exp <= 9 :
        score += 15
    elif exp >= 4:
        score += 8

    # Skills (15)
    skill_names = {s["name"] for s in skills}
    matched = required_skills.intersection(skill_names)
    score += min(len(matched) * 2, 15)

    # Career (10)
    career_text = " ".join(
        job["description"] for job in career
    ).lower()

    hits = sum(
        keyword in career_text
        for keyword in career_keywords
    )

    score += min(hits * 2, 10)

    # Behavioral (10)
    if signals["open_to_work_flag"]:
        score += 2

    if signals["notice_period_days"] <= 30:
        score += 2

    if signals["github_activity_score"] >= 7:
        score += 3

    if signals["recruiter_response_rate"] >= 0.30:
        score += 3

    return score

print("Rule scoring function loaded successfully!")




# ------------------------------------------------
# Batch Processing Setup
# ------------------------------------------------

results = []

BATCH_SIZE = 256

documents = []
candidates = []

print(f"Batch size: {BATCH_SIZE}")

with open("candidates.jsonl", "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        # Only process the first 1000 candidates for testing
        if i == 1000:
            break

        candidate = json.loads(line)

        candidates.append(candidate)
        documents.append(build_candidate_document(candidate))

        # Process a full batch
        if len(documents) == BATCH_SIZE:

            candidate_embeddings = model.encode(
                documents,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            similarities = cosine_similarity(
                candidate_embeddings,
                job_embedding.reshape(1, -1)
            ).flatten()

            for cand, similarity in zip(candidates, similarities):

                semantic_score = similarity * 60
                business_score = rule_score(cand)
                final_score = semantic_score + business_score

                results.append({
                    "candidate_id": cand["candidate_id"],
                    "score": round(float(final_score), 2),
                    "semantic": round(float(similarity), 4),
                    "rule": business_score
                })

            # Clear memory for next batch
            candidates.clear()
            documents.clear()

# ------------------------------------------------
# Process the last remaining batch
# ------------------------------------------------

if documents:

    print(f"Processing final batch of {len(documents)} candidates...")

    candidate_embeddings = model.encode(
        documents,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    similarities = cosine_similarity(
        candidate_embeddings,
        job_embedding.reshape(1, -1)
    ).flatten()

    for cand, similarity in zip(candidates, similarities):

        semantic_score = similarity * 60
        business_score = rule_score(cand)
        final_score = semantic_score + business_score

        results.append({
            "candidate_id": cand["candidate_id"],
            "score": round(float(final_score), 2),
            "semantic": round(float(similarity), 4),
            "rule": business_score
        })

print(f"\nTotal Candidates Processed: {len(results)}")


# ------------------------------------------------
# Sort Candidates
# ------------------------------------------------

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

print("\nTop 10 Candidates\n")

for rank, candidate in enumerate(results[:10], start=1):

    print(
        f"{rank}. "
        f"{candidate['candidate_id']} | "
        f"Score: {candidate['score']} | "
        f"Semantic: {candidate['semantic']} | "
        f"Rule: {candidate['rule']}"
    )

    # ====================================================
# CREATE SUBMISSION CSV
# ====================================================

import csv
import os

os.makedirs("output", exist_ok=True)

with open(
    "output/submission.csv",
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "candidate_id",
        "rank",
        "score",
        "reasoning"
    ])

    for rank, row in enumerate(results[:100], start=1):

        reasoning = (
            f"Strong semantic match ({row['semantic']:.4f}), "
            f"Business score {row['rule']}"
        )

        writer.writerow([
            row["candidate_id"],
            rank,
            row["score"],
            reasoning
        ])

print("\n✅ Submission file generated successfully!")
print("📁 File saved at: output/submission.csv")
# ------------------------------------------------
# Generate Batch Embeddings
# ------------------------------------------------

print("\nGenerating candidate embeddings...")

candidate_embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Candidate Embeddings Shape:", candidate_embeddings.shape)


# ------------------------------------------------
# Compute Semantic Similarity
# ------------------------------------------------

print("\nComputing semantic similarity...")

similarities = cosine_similarity(
    candidate_embeddings,
    job_embedding.reshape(1, -1)
).flatten()

print(f"Similarity scores generated: {len(similarities)}")

print("\nTop 5 Similarity Scores:")

for i in range(5):
    print(f"{candidates[i]['candidate_id']} -> {similarities[i]:.4f}")


    # ------------------------------------------------
# Final Hybrid Score
# ------------------------------------------------

print("\nCalculating hybrid scores...")

results = []

for candidate, similarity in zip(candidates, similarities):

    semantic_score = similarity * 60

    business_score = rule_score(candidate)

    final_score = semantic_score + business_score

    results.append({
        "candidate_id": candidate["candidate_id"],
        "semantic_score": round(similarity, 4),
        "rule_score": business_score,
        "final_score": round(final_score, 2)
    })

# Sort by final score
results.sort(
    key=lambda x: x["final_score"],
    reverse=True
)

print("\nTop 10 Ranked Candidates\n")

for rank, row in enumerate(results[:10], start=1):

    print(
        f"{rank}. "
        f"{row['candidate_id']} | "
        f"Final: {row['final_score']} | "
        f"Semantic: {row['semantic_score']} | "
        f"Rule: {row['rule_score']}"
    )


