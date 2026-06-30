import json

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    candidate = json.loads(next(f))

profile = candidate["profile"]
career = candidate["career_history"]
skills = candidate["skills"]
signals = candidate["redrob_signals"]

features = {}

# Basic Profile
features["candidate_id"] = candidate["candidate_id"]
features["experience_years"] = profile["years_of_experience"]
features["current_title"] = profile["current_title"]
features["current_company"] = profile["current_company"]
features["location"] = profile["location"]

# Skills
features["skill_names"] = [skill["name"] for skill in skills]

# Career History
features["companies"] = [job["company"] for job in career]
features["titles"] = [job["title"] for job in career]

# Behavioral Signals
features["open_to_work"] = signals["open_to_work_flag"]
features["notice_period"] = signals["notice_period_days"]
features["github_score"] = signals["github_activity_score"]
features["response_rate"] = signals["recruiter_response_rate"]
features["last_active"] = signals["last_active_date"]

print("=" * 80)
print("EXTRACTED FEATURES")
print("=" * 80)

for key, value in features.items():
    print(f"{key}: {value}")