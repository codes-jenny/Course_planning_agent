"""
eval/run_eval.py
Runs all 25 test queries through the pipeline and produces an evaluation report.

Run with:
    python eval/run_eval.py
"""

import sys
import json
import time
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import run_pipeline

QUERIES_FILE = Path(__file__).resolve().parent / "test_queries.json"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def has_citation(response: str) -> bool:
    """Check if the response contains at least one citation."""
    return "Source:" in response or "source:" in response.lower()


def is_abstention(response: str) -> bool:
    """Check if the response is a proper abstention."""
    markers = [
        "I don't have that information",
        "not in the provided catalog",
        "not in catalog",
    ]
    return any(m.lower() in response.lower() for m in markers)


def evaluate_query(query_obj: dict) -> dict:
    """Run one query and evaluate it."""
    start = time.time()

    result = run_pipeline(query_obj["query"], verbose=False)
    elapsed = round(time.time() - start, 2)

    response    = result["final_response"]
    category    = query_obj["category"]
    expected    = query_obj["expected_decision"]
    cited       = has_citation(response)
    abstained   = is_abstention(response)
    verdict     = result.get("verdict", "N/A")

    # Score correctness
    correct = None
    if category == "not_in_docs":
        correct = abstained   # must abstain
    elif category in ("prereq_check", "prereq_chain", "program_requirement", "plan"):
        correct = cited       # must have citations (manual grading for actual decision)

    return {
        "id":             query_obj["id"],
        "category":       category,
        "query":          query_obj["query"][:80] + "...",
        "expected":       expected,
        "verdict":        verdict,
        "has_citation":   cited,
        "is_abstention":  abstained,
        "auto_correct":   correct,
        "latency_s":      elapsed,
        "response":       response,
        "notes":          query_obj.get("notes", ""),
    }


def run_evaluation():
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    print(f"Running {len(queries)} test queries...\n")
    results = []

    for q in queries:
        print(f"  [{q['id']:02d}/{len(queries)}] {q['category']} — {q['query'][:60]}...")
        try:
            result = evaluate_query(q)
        except Exception as e:
            result = {
                "id":           q["id"],
                "category":     q["category"],
                "query":        q["query"][:80],
                "expected":     q["expected_decision"],
                "verdict":      "ERROR",
                "has_citation": False,
                "is_abstention":False,
                "auto_correct": False,
                "latency_s":    0,
                "response":     f"ERROR: {str(e)}",
                "notes":        q.get("notes", ""),
            }
        results.append(result)
        time.sleep(0.5)   # avoid rate limits

    # ── Metrics ───────────────────────────────────────────────────────────────
    total          = len(results)
    cited          = sum(1 for r in results if r["has_citation"])
    abstentions_ok = sum(1 for r in results if r["category"] == "not_in_docs" and r["is_abstention"])
    not_in_docs    = sum(1 for r in results if r["category"] == "not_in_docs")
    auto_correct   = sum(1 for r in results if r["auto_correct"] is True)
    auto_scored    = sum(1 for r in results if r["auto_correct"] is not None)

    citation_rate   = round(cited / total * 100, 1)
    abstention_rate = round(abstentions_ok / not_in_docs * 100, 1) if not_in_docs else 0
    auto_acc        = round(auto_correct / auto_scored * 100, 1) if auto_scored else 0
    avg_latency     = round(sum(r["latency_s"] for r in results) / total, 2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries          : {total}")
    print(f"Citation coverage      : {cited}/{total} ({citation_rate}%)")
    print(f"Abstention accuracy    : {abstentions_ok}/{not_in_docs} ({abstention_rate}%)")
    print(f"Auto-scored correct    : {auto_correct}/{auto_scored} ({auto_acc}%)")
    print(f"Average latency        : {avg_latency}s")
    print("="*60)

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — full detail
    json_path = RESULTS_DIR / f"eval_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "total":            total,
                "citation_rate":    citation_rate,
                "abstention_rate":  abstention_rate,
                "auto_accuracy":    auto_acc,
                "avg_latency_s":    avg_latency,
            },
            "results": results,
        }, f, indent=2)

    # CSV — easy to open in Excel/Sheets for manual grading
    csv_path = RESULTS_DIR / f"eval_results_{ts}.csv"
    csv_fields = ["id","category","query","expected","verdict",
                  "has_citation","is_abstention","auto_correct","latency_s","notes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})

    print(f"\nDetailed results → {json_path}")
    print(f"CSV for grading  → {csv_path}")

    # ── Print 3 example transcripts ──────────────────────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE TRANSCRIPTS")
    print("="*60)

    prereq_pass = next((r for r in results if r["category"] == "prereq_check" and r["has_citation"]), None)
    plan_result = next((r for r in results if r["category"] == "plan"), None)
    abstain_ex  = next((r for r in results if r["category"] == "not_in_docs" and r["is_abstention"]), None)

    for label, ex in [
        ("1. Correct eligibility decision with citations", prereq_pass),
        ("2. Course plan with justification + citations",  plan_result),
        ("3. Correct abstention + guidance",               abstain_ex),
    ]:
        print(f"\n--- {label} ---")
        if ex:
            print(f"Query: {ex['query']}")
            print(f"Response:\n{ex['response'][:600]}...")
        else:
            print("(No matching result found)")


if __name__ == "__main__":
    run_evaluation()