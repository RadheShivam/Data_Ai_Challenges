def build_candidate_document(candidate):
    profile = candidate["profile"]

    career = "\n".join(
        job["description"]
        for job in candidate["career_history"]
    )

    skills = ", ".join(
        skill["name"]
        for skill in candidate["skills"]
    )

    education = ", ".join(
        f"{edu['degree']} in {edu['field_of_study']}"
        for edu in candidate["education"]
    )

    languages = ", ".join(
        lang["language"]
        for lang in candidate["languages"]
    )

    return f"""
Headline:
{profile['headline']}

Summary:
{profile['summary']}

Experience:
{profile['years_of_experience']} years

Current Title:
{profile['current_title']}

Career History:
{career}

Skills:
{skills}

Education:
{education}

Languages:
{languages}
"""