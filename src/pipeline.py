"""
src/pipeline.py
Wires the 4 chains together into a single end-to-end function.

Flow:
  student message
      → IntakeChain   (parse profile, ask clarifying Qs if needed)
      → RetrieverChain (semantic search, build context)
      → PlannerChain  (generate cited plan)
      → VerifierChain (audit, revise if needed)
      → final response
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chains.intake    import run_intake, is_profile_complete, format_clarifying_questions
from chains.retriever import retrieve_multi, format_context
from chains.planner   import run_planner
from chains.verifier  import run_verifier


def run_pipeline(student_message: str, verbose: bool = False) -> dict:
    """
    Full end-to-end pipeline.

    Returns:
    {
        "profile":        dict,
        "clarifying_qs":  str,   # non-empty if profile incomplete
        "context_chunks": list,
        "plan":           str,
        "verdict":        str,
        "final_response": str,
        "needs_more_info": bool,
    }
    """

    # ── STEP 1: Intake ────────────────────────────────────────────────────────
    if verbose:
        print("[ INTAKE ] Parsing student message...")

    profile = run_intake(student_message)

    if verbose:
        print(json.dumps(profile, indent=2))

    # If profile is incomplete, return clarifying questions immediately
    if not is_profile_complete(profile):
        clarifying = format_clarifying_questions(profile)
        return {
            "profile":         profile,
            "clarifying_qs":   clarifying,
            "context_chunks":  [],
            "plan":            "",
            "verdict":         "INCOMPLETE",
            "final_response":  clarifying,
            "needs_more_info": True,
        }

    # ── STEP 2: Build search queries ─────────────────────────────────────────
    if verbose:
        print("[ RETRIEVER ] Building search queries...")

    # Always search for:
    #   1. The student's actual question
    #   2. CSE minor requirements
    #   3. Each completed course's prereq info
    completed = profile.get("completed_courses", [])
    queries = [
        student_message,
        f"CSE minor requirements prerequisites credits IIT Kharagpur",
        f"minor registration eligibility CGPA backlog policy",
    ]
    # Add targeted queries for courses the student has completed
    for course in completed[:5]:   # cap at 5 to avoid too many queries
        queries.append(f"{course} prerequisite credits")

    chunks = retrieve_multi(queries, k=5)
    context = format_context(chunks)

    if verbose:
        print(f"[ RETRIEVER ] Retrieved {len(chunks)} unique chunks.")

    # ── STEP 3: Planner ───────────────────────────────────────────────────────
    if verbose:
        print("[ PLANNER ] Generating course plan...")

    plan = run_planner(
        student_profile=profile,
        context=context,
        student_question=student_message,
    )

    if verbose:
        print("[ PLANNER ] Plan generated.")

    # ── STEP 4: Verifier ──────────────────────────────────────────────────────
    if verbose:
        print("[ VERIFIER ] Auditing plan...")

    verification = run_verifier(plan=plan, context=context)

    if verbose:
        print(f"[ VERIFIER ] Verdict: {verification['verdict']}")
        if verification["issues"] and verification["issues"] != "None":
            print(f"[ VERIFIER ] Issues: {verification['issues']}")

    final_response = verification["revised_plan"]

    return {
        "profile":         profile,
        "clarifying_qs":   "",
        "context_chunks":  chunks,
        "plan":            plan,
        "verdict":         verification["verdict"],
        "issues":          verification["issues"],
        "final_response":  final_response,
        "needs_more_info": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI MODE — for quick testing without the UI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("IIT KGP Course Planning Assistant (CLI)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        result = run_pipeline(user_input, verbose=True)
        print("\n" + "="*70)
        print(result["final_response"])
        print("="*70 + "\n")