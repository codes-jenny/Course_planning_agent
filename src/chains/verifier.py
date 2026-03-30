"""
src/chains/verifier.py
Audits the planner's output against the retrieved context.
Strips or corrects any claims that are not grounded in the catalog.
"""

from openai import OpenAI
from dotenv import load_dotenv
from prompts import VERIFIER_PROMPT

load_dotenv()
client = OpenAI()


def run_verifier(plan: str, context: str) -> dict:
    """
    Audits the plan against the catalog context.

    Returns:
    {
        "verdict":      "PASS" | "FAIL" | "NEEDS REVISION",
        "issues":       str,
        "revised_plan": str,   # same as plan if PASS
        "raw_output":   str,
    }
    """
    # Skip verification if plan is already an abstention
    if "I don't have that information" in plan and "ANSWER / PLAN" in plan:
        return {
            "verdict":      "PASS",
            "issues":       "None — abstention response, no claims to verify.",
            "revised_plan": plan,
            "raw_output":   plan,
        }

    prompt = VERIFIER_PROMPT.format(context=context, plan=plan)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Parse verdict
    verdict = "NEEDS REVISION"
    if "VERDICT: PASS" in raw:
        verdict = "PASS"
    elif "VERDICT: FAIL" in raw:
        verdict = "FAIL"

    # Extract issues section
    issues = _extract_section(raw, "ISSUES FOUND:", "REVISED PLAN")
    if not issues:
        issues = _extract_section(raw, "ISSUES FOUND:", "FINAL CITATIONS")

    # Use revised plan if provided, otherwise keep original
    revised = _extract_section(raw, "REVISED PLAN", "FINAL CITATIONS")
    final_plan = revised.strip() if revised and revised.strip() not in ("", "None") else plan

    return {
        "verdict":      verdict,
        "issues":       issues.strip(),
        "revised_plan": final_plan,
        "raw_output":   raw,
    }


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extracts text between two section markers.
    """
    try:
        start = text.index(start_marker) + len(start_marker)
        end   = text.index(end_marker, start)
        return text[start:end].strip()
    except ValueError:
        return ""


if __name__ == "__main__":
    # Quick test
    dummy_plan = """
---
ANSWER / PLAN:
Student is eligible. Recommended: CS31001 Operating Systems, CS31005 Algorithms-II.

WHY:
CS31001 requires CS31007 which student has completed.
CS31005 has no prerequisites.

CITATIONS:
Source: cs_courses.json, Section: CS31001
Source: cs_courses.json, Section: CS31005

CLARIFYING QUESTIONS (if needed):
None

ASSUMPTIONS / NOT IN CATALOG:
Course availability in Spring 2025 not confirmed in catalog.
---
"""
    dummy_context = (
        "[Source: cs_courses.json | Section: CS31001 | Score: 0.95]\n"
        "Course Code: CS31001\nPrerequisites: CS31007 Computer Organization\n\n"
        "---\n\n"
        "[Source: cs_courses.json | Section: CS31005 | Score: 0.92]\n"
        "Course Code: CS31005\nPrerequisites: None"
    )
    result = run_verifier(dummy_plan, dummy_context)
    print(f"Verdict: {result['verdict']}")
    print(f"Issues: {result['issues']}")
    print(f"Final plan:\n{result['revised_plan']}")