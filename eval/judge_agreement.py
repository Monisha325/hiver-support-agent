"""
Phase 7 — Judge vs human agreement study.

Computes Cohen's kappa between:
  - LLM judge scores (from eval/judge_results.json)
  - Human scores (from eval/human_scores.csv — you fill this in)

INSTRUCTIONS FOR HUMAN SCORING:
  1. Open eval/judge_results.json
  2. For each message_id, read the agent result in eval/agent_results.json
  3. Score on the same 4 dimensions (0-3 each) using the rubric in
     eval/judge_rubric.txt
  4. Enter your scores in eval/human_scores.csv with columns:
       message_id, groundedness, accuracy, helpfulness, tone, total
  5. Run this script to compute agreement

The script computes Cohen's kappa on:
  - Per-dimension: groundedness, accuracy, helpfulness, tone
  - Total score (binned: 0-4=low, 5-8=medium, 9-12=high)

Kappa interpretation:
  < 0.20  = slight agreement
  0.21–0.40 = fair
  0.41–0.60 = moderate
  0.61–0.80 = substantial
  > 0.80  = near-perfect

Usage:
    python eval/judge_agreement.py
"""

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_JSON    = ROOT / "eval" / "judge_results.json"
HUMAN_CSV     = ROOT / "eval" / "human_scores.csv"
AGREEMENT_TXT = ROOT / "eval" / "agreement_report.txt"

# Sample of message IDs to score (first 20 judged examples) — human scores these
SAMPLE_N = 20


def cohen_kappa(rater1_scores, rater2_scores, n_categories=None):
    """
    Compute Cohen's kappa for two lists of ordinal/categorical scores.
    For ordinal data (0-3), uses linear-weighted kappa.
    """
    from sklearn.metrics import cohen_kappa_score
    # linear weights for 0-3 scale
    return cohen_kappa_score(rater1_scores, rater2_scores, weights="linear")


def bin_total(score):
    """Bin total 0-12 into low/medium/high for categorical kappa."""
    if score <= 4:  return "low"
    if score <= 8:  return "medium"
    return "high"


def generate_scoring_sheet():
    """Write a scoring sheet CSV pre-populated with message IDs."""
    if not JUDGE_JSON.exists():
        sys.exit(f"ERROR: {JUDGE_JSON} not found. Run llm_judge.py first.")

    with open(JUDGE_JSON, encoding="utf-8") as f:
        judge_results = json.load(f)

    valid = [j for j in judge_results if j.get("total", -1) >= 0][:SAMPLE_N]

    sheet_path = ROOT / "eval" / "human_scores_template.csv"
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "message_id", "groundedness", "accuracy", "helpfulness", "tone", "total", "notes"
        ])
        writer.writeheader()
        for j in valid:
            writer.writerow({
                "message_id":    j["message_id"],
                "groundedness":  "",   # FILL IN: 0-3
                "accuracy":      "",   # FILL IN: 0-3
                "helpfulness":   "",   # FILL IN: 0-3
                "tone":          "",   # FILL IN: 0-3
                "total":         "",   # FILL IN: sum of above
                "notes":         "",   # optional
            })

    print(f"[INFO] Scoring template written -> {sheet_path}")
    print(f"[INFO] Fill in {len(valid)} rows, save as eval/human_scores.csv, then re-run.")
    return valid


def compute_agreement():
    if not JUDGE_JSON.exists():
        sys.exit(f"ERROR: {JUDGE_JSON} not found.")
    if not HUMAN_CSV.exists():
        sys.exit(
            f"ERROR: {HUMAN_CSV} not found.\n"
            "Run: python eval/judge_agreement.py  (to generate template)\n"
            "Then fill in human_scores_template.csv, save as human_scores.csv."
        )

    with open(JUDGE_JSON, encoding="utf-8") as f:
        judge_data = {j["message_id"]: j for j in json.load(f)}

    with open(HUMAN_CSV, encoding="utf-8") as f:
        human_data = {r["message_id"]: r for r in csv.DictReader(f)}

    # Align on shared message IDs
    common_ids = [mid for mid in human_data if mid in judge_data]
    if len(common_ids) < 5:
        sys.exit(f"ERROR: Only {len(common_ids)} common IDs. Need at least 5.")

    dims = ["groundedness", "accuracy", "helpfulness", "tone"]
    results = {}

    for dim in dims:
        llm_scores   = [int(judge_data[mid][dim]) for mid in common_ids]
        human_scores = [int(human_data[mid][dim]) for mid in common_ids]
        kappa = cohen_kappa(llm_scores, human_scores)
        results[dim] = {
            "kappa":        kappa,
            "llm_mean":     sum(llm_scores)   / len(llm_scores),
            "human_mean":   sum(human_scores) / len(human_scores),
        }

    # Total binned kappa
    llm_bins   = [bin_total(int(judge_data[mid]["total"])) for mid in common_ids]
    human_bins = [bin_total(int(human_data[mid]["total"])) for mid in common_ids]
    from sklearn.metrics import cohen_kappa_score
    total_kappa = cohen_kappa_score(llm_bins, human_bins)

    # Format report
    lines = [
        "=" * 60,
        "JUDGE vs HUMAN AGREEMENT REPORT",
        f"N compared: {len(common_ids)}",
        f"Agreement metric: Cohen's kappa (linear weights for 0-3 dims)",
        "=" * 60, "",
        "Per-dimension (linear-weighted kappa):",
    ]
    for dim, r in results.items():
        interp = (
            "near-perfect" if r["kappa"] > 0.80 else
            "substantial"  if r["kappa"] > 0.60 else
            "moderate"     if r["kappa"] > 0.40 else
            "fair"         if r["kappa"] > 0.20 else
            "slight"
        )
        lines.append(
            f"  {dim:<15} kappa={r['kappa']:+.3f}  ({interp})  "
            f"llm_mean={r['llm_mean']:.2f}  human_mean={r['human_mean']:.2f}"
        )

    lines += [
        "",
        f"Overall total (binned low/medium/high): kappa={total_kappa:+.3f}",
        "",
        "NOTE: These numbers are reported as-is, weak or strong.",
        "The kappa is computed over the judge subset (N<=50 examples).",
    ]

    report = "\n".join(lines)
    print(report)
    AGREEMENT_TXT.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Agreement report saved -> {AGREEMENT_TXT}")
    return results


if __name__ == "__main__":
    if not JUDGE_JSON.exists():
        print("[INFO] eval/judge_results.json not found.")
        print("       Run eval/llm_judge.py first (requires API key), then re-run.")
        print("       Scoring template will be generated once judge results exist.")
    elif not HUMAN_CSV.exists():
        print("[INFO] human_scores.csv not found — generating scoring template ...")
        generate_scoring_sheet()
    else:
        compute_agreement()
