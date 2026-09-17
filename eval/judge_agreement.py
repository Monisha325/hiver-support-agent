"""
Phase 7 — Judge vs human agreement study.

Computes Cohen's kappa between:
  - LLM judge scores  (eval/judge_results.json)
  - Human scores      (eval/human_scores.csv — you fill this in)
joined by message_id.

GUARDRAIL: If eval/human_scores.csv does not exist this script exits
immediately. It never creates, mocks, or synthesises scores.

Usage:
    python eval/judge_agreement.py          # compute kappa (needs human_scores.csv)
    python eval/judge_agreement.py --sheet  # regenerate blind scoring template

Kappa bands (Landis & Koch, 1977):
  < 0.20  = slight
  0.21–0.40 = fair
  0.41–0.60 = moderate
  0.61–0.80 = substantial
  > 0.80  = almost perfect
"""

import csv
import json
import pathlib
import sys
import re

ROOT          = pathlib.Path(__file__).parent.parent
JUDGE_JSON    = ROOT / "eval" / "judge_results.json"
RESULTS_JSON  = ROOT / "eval" / "agent_results.json"
HUMAN_CSV     = ROOT / "eval" / "human_scores.csv"
AGREEMENT_TXT = ROOT / "eval" / "agreement_report.txt"
TEMPLATE_CSV  = ROOT / "eval" / "human_scores_template.csv"
README        = ROOT / "README.md"
REPORT        = ROOT / "report" / "REPORT.md"

DIMS = ["groundedness", "accuracy", "helpfulness", "tone"]
SAMPLE_N = 20


# ── helpers ───────────────────────────────────────────────────────────────

def kappa_band(k):
    if k > 0.80:  return "almost perfect"
    if k > 0.60:  return "substantial"
    if k > 0.40:  return "moderate"
    if k > 0.20:  return "fair"
    return "slight"


def raw_agreement(a, b):
    """Fraction of pairs where scores are identical."""
    assert len(a) == len(b)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def linear_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b, weights="linear")


def bin_total(score):
    if score <= 4:  return "low"
    if score <= 8:  return "medium"
    return "high"


# ── Step 1: generate blind scoring template ────────────────────────────────

def generate_scoring_sheet():
    """Write a blind scoring sheet (no judge scores visible)."""
    import random

    for path, name in [(JUDGE_JSON, "judge_results.json"), (RESULTS_JSON, "agent_results.json")]:
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run the appropriate pipeline step first.")

    with open(JUDGE_JSON, encoding="utf-8") as f:
        judge_results = json.load(f)
    with open(RESULTS_JSON, encoding="utf-8") as f:
        agent_results = {r["message_id"]: r for r in json.load(f)}

    valid = [j for j in judge_results if j.get("total", -1) >= 0][:SAMPLE_N]
    random.seed(42)        # reproducible shuffle
    random.shuffle(valid)

    fields = [
        "message_id", "customer_message", "classified_intent",
        "retrieved_references", "draft_reply",
        "escalation_decision", "escalation_reason",
        "groundedness", "accuracy", "helpfulness", "tone", "total", "notes"
    ]
    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j in valid:
            mid = j["message_id"]
            ar  = agent_results.get(mid, {})
            refs = "\n---\n".join(
                h.get("reply_text", "") for h in ar.get("retrieved_hits", [])
                if isinstance(h, dict)
            )
            writer.writerow({
                "message_id":          mid,
                "customer_message":    ar.get("input_text", ""),
                "classified_intent":   ar.get("pred_intent", ""),
                "retrieved_references": refs,
                "draft_reply":         ar.get("draft_reply", ""),
                "escalation_decision": ar.get("pred_decision", ""),
                "escalation_reason":   ar.get("pred_reason", ""),
                "groundedness": "",   # FILL IN 0-3
                "accuracy":     "",   # FILL IN 0-3
                "helpfulness":  "",   # FILL IN 0-3
                "tone":         "",   # FILL IN 0-3
                "total":        "",   # FILL IN (sum of above)
                "notes":        "",   # optional free text
            })

    n = len(valid)
    print(f"[INFO] Scoring template written -> {TEMPLATE_CSV}")
    print(f"[INFO] {n} examples | estimated time: {n}–{n*2} minutes")
    print(f"[INFO] Fill in the score columns, save as eval/human_scores.csv, then re-run without --sheet")


# ── Step 2: validate human_scores.csv ─────────────────────────────────────

def validate_human_scores(human_data, judge_data):
    errors = []

    # Check IDs align
    human_ids = set(human_data.keys())
    judge_ids  = set(judge_data.keys())
    missing_from_judge = human_ids - judge_ids
    if missing_from_judge:
        errors.append(f"IDs in human_scores.csv not found in judge_results.json: {missing_from_judge}")

    for mid, row in human_data.items():
        for dim in DIMS + ["total"]:
            val = row.get(dim, "").strip()
            if val == "":
                errors.append(f"[{mid}] '{dim}' is empty")
                continue
            try:
                v = int(val)
            except ValueError:
                errors.append(f"[{mid}] '{dim}' = '{val}' is not an integer")
                continue
            if dim == "total":
                if not (0 <= v <= 12):
                    errors.append(f"[{mid}] 'total' = {v} out of range 0-12")
            else:
                if not (0 <= v <= 3):
                    errors.append(f"[{mid}] '{dim}' = {v} out of range 0-3")

    if errors:
        print("\n[VALIDATION ERRORS]")
        for e in errors:
            print(f"  {e}")
        sys.exit("Fix the above errors in human_scores.csv and re-run.")

    print(f"[INFO] Validation passed: {len(human_data)} rows, all scores in range.")


# ── Step 3: compute agreement ──────────────────────────────────────────────

def compute_agreement():
    # Hard guardrail — do not proceed without real human file
    if not HUMAN_CSV.exists():
        sys.exit(
            f"\nSTOPPED: {HUMAN_CSV} does not exist.\n"
            "Fill in eval/human_scores_template.csv, save as eval/human_scores.csv, then re-run.\n"
            "Do NOT create a placeholder or synthetic version of this file."
        )

    with open(JUDGE_JSON, encoding="utf-8") as f:
        judge_data = {j["message_id"]: j for j in json.load(f)}
    with open(HUMAN_CSV, encoding="utf-8") as f:
        human_data = {r["message_id"]: r for r in csv.DictReader(f)}
    with open(RESULTS_JSON, encoding="utf-8") as f:
        agent_data = {r["message_id"]: r for r in json.load(f)}

    validate_human_scores(human_data, judge_data)

    common_ids = sorted(set(human_data.keys()) & set(judge_data.keys()))
    n = len(common_ids)

    # Per-dimension kappa + raw agreement
    dim_results = {}
    for dim in DIMS:
        llm_s   = [int(judge_data[mid][dim])      for mid in common_ids]
        human_s = [int(human_data[mid][dim].strip()) for mid in common_ids]
        k    = linear_kappa(llm_s, human_s)
        agrm = raw_agreement(llm_s, human_s)
        dim_results[dim] = {
            "kappa":         k,
            "raw_agreement": agrm,
            "llm_mean":      sum(llm_s)   / n,
            "human_mean":    sum(human_s) / n,
            "llm_scores":    llm_s,
            "human_scores":  human_s,
        }

    # Overall kappa: treat each (example, dimension) pair as one rating
    # This is a common aggregation when per-dim samples are small
    all_llm   = [s for dim in DIMS for s in dim_results[dim]["llm_scores"]]
    all_human = [s for dim in DIMS for s in dim_results[dim]["human_scores"]]
    from sklearn.metrics import cohen_kappa_score
    overall_kappa    = linear_kappa(all_llm, all_human)
    overall_agrm     = raw_agreement(all_llm, all_human)

    # Also binned-total kappa
    llm_bins   = [bin_total(int(judge_data[mid]["total"])) for mid in common_ids]
    human_bins = [bin_total(int(human_data[mid]["total"].strip())) for mid in common_ids]
    # Only compute if all three bins are present (kappa undefined with only one class)
    if len(set(llm_bins) | set(human_bins)) > 1:
        total_kappa = cohen_kappa_score(llm_bins, human_bins)
    else:
        total_kappa = float("nan")

    # ── Disagreement table (>1 point diff on any dim) ─────────────────────
    disagreements = []
    for mid in common_ids:
        for dim in DIMS:
            llm_v   = int(judge_data[mid][dim])
            human_v = int(human_data[mid][dim].strip())
            if abs(llm_v - human_v) > 1:
                ar = agent_data.get(mid, {})
                disagreements.append({
                    "message_id":     mid,
                    "dimension":      dim,
                    "human_score":    human_v,
                    "judge_score":    llm_v,
                    "diff":          llm_v - human_v,
                    "customer_msg":  ar.get("input_text", "")[:80],
                    "draft_reply":   ar.get("draft_reply", "")[:80],
                })

    # ── Format report ────────────────────────────────────────────────────
    sep = "=" * 68
    lines = [
        sep,
        "JUDGE vs HUMAN AGREEMENT REPORT",
        f"N examples: {n}  |  Dimensions: {', '.join(DIMS)}",
        f"Method: linear-weighted Cohen's kappa (Landis & Koch bands)",
        sep, "",
        f"{'Dimension':<15}  {'Kappa':>7}  {'Band':<15}  {'Raw Agree':>10}  {'LLM mean':>9}  {'Human mean':>10}",
        "-" * 68,
    ]
    for dim, r in dim_results.items():
        lines.append(
            f"{dim:<15}  {r['kappa']:>+7.3f}  {kappa_band(r['kappa']):<15}  "
            f"{r['raw_agreement']:>9.1%}  {r['llm_mean']:>9.2f}  {r['human_mean']:>10.2f}"
        )
    lines += [
        "-" * 68,
        f"{'OVERALL (all dims)':<15}  {overall_kappa:>+7.3f}  {kappa_band(overall_kappa):<15}  {overall_agrm:>9.1%}",
        f"{'Total (binned)':<15}  {total_kappa:>+7.3f}",
        "",
        "NOTE: Kappa can appear low on n<30 samples even with decent raw agreement.",
        "      Raw agreement % is provided as a complementary sanity check.",
        "",
    ]

    if disagreements:
        lines += [
            f"LARGE DISAGREEMENTS (|human - judge| > 1):  {len(disagreements)} found",
            f"{'ID':<12}  {'Dim':<15}  {'Human':>6}  {'Judge':>6}  {'Diff':>5}  Customer message",
            "-" * 68,
        ]
        for d in disagreements:
            lines.append(
                f"{d['message_id']:<12}  {d['dimension']:<15}  {d['human_score']:>6}  "
                f"{d['judge_score']:>6}  {d['diff']:>+5}  {d['customer_msg']}"
            )
    else:
        lines.append("LARGE DISAGREEMENTS (>1 point): none")

    report = "\n".join(lines)
    print("\n" + report)
    AGREEMENT_TXT.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Agreement report saved -> {AGREEMENT_TXT}")

    return {
        "n":              n,
        "dim_results":    dim_results,
        "overall_kappa":  overall_kappa,
        "overall_agrm":   overall_agrm,
        "total_kappa":    total_kappa,
        "disagreements":  disagreements,
    }


# ── Step 4: update README and REPORT.md ───────────────────────────────────

def update_docs(stats):
    n             = stats["n"]
    ok            = stats["overall_kappa"]
    agrm          = stats["overall_agrm"]
    band          = kappa_band(ok)
    dim_results   = stats["dim_results"]
    disagree_n    = len(stats["disagreements"])

    # ── README: replace the Kappa row ────────────────────────────────────
    readme_text = README.read_text(encoding="utf-8")
    new_kappa_row = (
        f"| Kappa (judge vs human) | **{ok:+.3f}** (overall, {band}) | "
        f"n={n}, raw agree {agrm:.0%}; per-dim: "
        + ", ".join(f"{d} {r['kappa']:+.3f}" for d, r in dim_results.items())
        + " |"
    )
    readme_text = re.sub(
        r"\| Kappa \(judge vs human\) \|.*?\|.*?\|",
        new_kappa_row,
        readme_text
    )
    README.write_text(readme_text, encoding="utf-8")
    print(f"[INFO] README.md updated (kappa row replaced)")

    # ── REPORT.md: replace Section 6 ────────────────────────────────────
    interp = (
        f"The overall linear-weighted Cohen's kappa across all four dimensions is **{ok:+.3f}** "
        f"(n={n} examples × 4 dimensions = {n*4} rated pairs), which falls in the **\"{band}\"** "
        f"range on the Landis & Koch (1977) scale. "
    )
    if ok < 0.20:
        trust = (
            "This is weak agreement, meaning the LLM judge's automated scores on the remaining "
            f"{43 - n} examples cannot be taken as a reliable proxy for human judgment without "
            "extensive spot-checking. The judge scores should be treated as approximate indicators only."
        )
    elif ok < 0.40:
        trust = (
            "This is fair agreement. The LLM judge provides a directionally useful signal, but "
            "individual scores should be interpreted with caution — roughly 1 in 5 examples may "
            "diverge meaningfully from a human evaluator's view."
        )
    elif ok < 0.60:
        trust = (
            "This is moderate agreement. The LLM judge is reasonably aligned with human judgment "
            "on most examples, but should not be treated as a substitute for human review on high-stakes decisions."
        )
    else:
        trust = (
            "This represents substantial to near-perfect agreement, suggesting the LLM judge is "
            "a reliable proxy for human judgment on this rubric."
        )

    disagree_note = (
        f" {disagree_n} examples had a score gap of >1 point on at least one dimension — "
        "see `eval/agreement_report.txt` for the full disagreement table."
        if disagree_n else ""
    )

    new_section6 = (
        f"## 6. LLM Judge vs Human Agreement\n"
        f"43 examples were evaluated using the LLM-as-a-judge (Qwen via Groq API) on a "
        f"4-dimension rubric (groundedness, accuracy, helpfulness, tone, each 0–3). "
        f"{n} of those were independently scored by a human using the same rubric, with the "
        f"LLM judge's scores hidden during human scoring.\n\n"
        f"| Dimension | Kappa | Band | Raw agreement |\n"
        f"|-----------|-------|------|---------------|\n"
    )
    for dim, r in dim_results.items():
        new_section6 += (
            f"| {dim} | {r['kappa']:+.3f} | {kappa_band(r['kappa'])} | {r['raw_agreement']:.0%} |\n"
        )
    new_section6 += (
        f"\n{interp}{trust}{disagree_note}\n"
    )

    report_text = REPORT.read_text(encoding="utf-8")
    report_text = re.sub(
        r"## 6\. LLM Judge vs Human Agreement.*?(?=## \d|\Z)",
        new_section6 + "\n",
        report_text,
        flags=re.DOTALL
    )
    REPORT.write_text(report_text, encoding="utf-8")
    print(f"[INFO] report/REPORT.md Section 6 updated")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--sheet" in sys.argv:
        if not JUDGE_JSON.exists():
            sys.exit(f"ERROR: {JUDGE_JSON} not found. Run eval/llm_judge.py first.")
        generate_scoring_sheet()
    elif not JUDGE_JSON.exists():
        sys.exit(f"ERROR: {JUDGE_JSON} not found. Run eval/llm_judge.py first.")
    elif not HUMAN_CSV.exists():
        print("[INFO] eval/human_scores.csv not found.")
        print("       Generating blind scoring template ...")
        generate_scoring_sheet()
    else:
        stats = compute_agreement()
        update_docs(stats)

