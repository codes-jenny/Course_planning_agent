"""
src/chains/planner.py
Generates a grounded course plan using retrieved context.
Enforces citation and abstention rules via the prompt.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from prompts import PLANNER_PROMPT, ABSTENTION_RESPONSE

load_dotenv()
client = OpenAI()


def run_planner(
    student_profile: dict,
    context: str,
    student_question: str,
) -> str:
    """
    Given a validated student profile, retrieved context, and the question,
    returns a structured plan string following the mandatory output format.
    """

    # Hard guardrail — abstain immediately if no context retrieved
    if not context or context.strip() == "":
        return ABSTENTION_RESPONSE

    # CGPA eligibility check before even calling the LLM
    cgpa = student_profile.get("cgpa")
    backlogs = student_profile.get("backlogs", "unknown")

    if cgpa is not None and float(cgpa) < 7.5:
        return (
            "---\n"
            "ANSWER / PLAN:\n"
            "The student is not eligible to register for a Minor. "
            "A minimum CGPA of 7.5 is required at the time of registration "
            "(Source: ug_policy.txt, Section: MINOR IN A DISCIPLINE).\n\n"
            "WHY (requirements/prereqs satisfied):\n"
            f"Student CGPA: {cgpa} — below the 7.5 threshold.\n\n"
            "CITATIONS:\n"
            "Source: ug_policy.txt, Section: MINOR IN A DISCIPLINE — "
            "'Only those students who have a CGPA of 7.5 or above, without any Backlog, "
            "will be permitted to register for a Minor.'\n\n"
            "CLARIFYING QUESTIONS (if needed):\n"
            "None\n\n"
            "ASSUMPTIONS / NOT IN CATALOG:\n"
            "None — policy is clearly stated.\n"
            "---"
        )

    if backlogs == "yes":
        return (
            "---\n"
            "ANSWER / PLAN:\n"
            "The student is not eligible to register for a Minor due to active backlogs. "
            "(Source: ug_policy.txt, Section: MINOR IN A DISCIPLINE)\n\n"
            "WHY (requirements/prereqs satisfied):\n"
            "Students must be backlog-free at the time of Minor registration.\n\n"
            "CITATIONS:\n"
            "Source: ug_policy.txt, Section: MINOR IN A DISCIPLINE — "
            "'Only those students who have a CGPA of 7.5 or above, without any Backlog, "
            "will be permitted to register for a Minor.'\n\n"
            "CLARIFYING QUESTIONS (if needed):\n"
            "None\n\n"
            "ASSUMPTIONS / NOT IN CATALOG:\n"
            "None\n"
            "---"
        )

    profile_str = json.dumps(student_profile, indent=2)

    prompt = PLANNER_PROMPT.format(
        student_profile=profile_str,
        context=context,
        student_question=student_question,
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick test with dummy data
    profile = {
        "department": "ISE",
        "current_semester": 5,
        "cgpa": 8.5,
        "completed_courses": ["IM21003", "IM21001", "MA20104", "CS21001"],
        "target_minor": "CSE Minor",
        "target_term": "Spring 2025",
        "max_courses": 3,
        "backlogs": "no",
    }
    dummy_context = (
        "[Source: minor_cse.pdf | Section: page_1 | Score: 0.91]\n"
        "CSE Minor requires 6 subjects totaling 18-24 credits. "
        "Core subjects include CS21003, CS31001, CS31005.\n\n"
        "---\n\n"
        "[Source: ug_policy.txt | Section: MINOR IN A DISCIPLINE | Score: 0.88]\n"
        "Students must have CGPA >= 7.5 and no backlogs to register for a Minor."
    )
    question = "What courses should I take next semester for the CSE minor?"
    plan = run_planner(profile, dummy_context, question)
    print(plan)