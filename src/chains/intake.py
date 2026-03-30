"""
src/chains/intake.py
Parses the student's message, extracts their profile,
and returns clarifying questions if information is missing.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from prompts import INTAKE_PROMPT

load_dotenv()
client = OpenAI()


def run_intake(student_message: str) -> dict:
    """
    Takes the raw student message.
    Returns a parsed profile dict with missing_fields and clarifying_questions.
    """
    prompt = INTAKE_PROMPT.format(student_message=student_message)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model adds extra text around JSON
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        profile = json.loads(match.group()) if match else {}

    return profile


def is_profile_complete(profile: dict) -> bool:
    """
    Returns True only if the planner has enough info to generate a plan.
    Minimum required: department, cgpa, completed_courses, target_minor.
    """
    required = ["department", "cgpa", "completed_courses", "target_minor"]
    return all(profile.get(f) is not None for f in required)


def format_clarifying_questions(profile: dict) -> str:
    """
    Formats clarifying questions as a numbered list for the UI.
    """
    questions = profile.get("clarifying_questions", [])
    if not questions:
        return ""
    lines = [f"{i+1}. {q}" for i, q in enumerate(questions)]
    return "Before I can plan your courses, I need a few more details:\n\n" + "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    test_msg = "I am an ISE student in 5th semester with CGPA 8.2. I want to do CSE minor. I have completed IM21003, IM21001, MA20104."
    profile = run_intake(test_msg)
    print(json.dumps(profile, indent=2))
    print("\nProfile complete?", is_profile_complete(profile))