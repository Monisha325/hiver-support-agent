"""
Phase 7 — Evaluation harness.

Runs the full agent on eval/golden_set.csv and reports:
  - Intent: accuracy, per-class F1, confusion matrix, worst intents
  - Escalation: precision/recall (asymmetric — false-auto vs false-escalate)
  - Reply groundedness: ROUGE-1 recall between draft and retrieved passages
  - Summary table printed + saved to eval/metrics_report.txt

Usage:
    python eval/metrics.py
"""

import csv
import json
import pathlib
import sys
import time
import collections

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_CSV   = ROOT / "eval" / "golden_set.csv"
RESULTS_JSON = ROOT / "eval" / "agent_results.json"
REPORT_TXT   = ROOT / "eval" / "metrics_report.txt"


# ── 1. Run agent on golden set (skip if results already cached) ────────────
def run_agent_on_golden(golden_rows: list[dict]) -> list[dict]:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from agent.agent import run_agent

    results = []
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            results = json.load(f)
            
    processed_ids = {r["message_id"] for r in results}

    total = len(golden_rows)
    for i, row in enumerate(golden_rows):
        if row["message_id"] in processed_ids:
            continue
            
        print(f"\r  Running agent [{i+1}/{total}] ...", end="", flush=True)
        try:
            res = run_agent(row["input_text"])
            results.append({
                "message_id":       row["message_id"],
                "gold_intent":      row["gold_intent"],
                "gold_decision":    row["gold_decision"],
                "pred_intent":      res["intent"],
                "pred_decision":    res["decision"],
                "pred_reason":      res["reason"],
                "draft_reply":      res["draft_reply"],
                "retrieved_hits":   res["retrieved_hits"],
                "confidence":       res["confidence"],
                "input_text":       row["input_text"],
                "reference_note":   row.get("reference_resolution_note", ""),
            })
        except Exception as e:
            print(f"\n[ERROR] API failed on {row['message_id']}: {e}")
            print("Stopping to allow resume later.")
            raise e
            
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        time.sleep(4) # Rate limit mitigation
            
    print()
    return results


# ── 2. Intent metrics ──────────────────────────────────────────────────────
def intent_metrics(results: list[dict]) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix
    )
    import numpy as np

    golds = [r["gold_intent"] for r in results if r["gold_intent"] not in ("UNCLASSIFIED", "OOS")]
    preds = [r["pred_intent"] for r in results if r["gold_intent"] not in ("UNCLASSIFIED", "OOS")]

    acc = accuracy_score(golds, preds)
    report = classification_report(golds, preds, output_dict=True, zero_division=0)

    # Worst-performing intents by F1
    class_f1 = {k: v["f1-score"] for k, v in report.items()
                if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")}
    worst = sorted(class_f1.items(), key=lambda x: x[1])[:3]

    labels = sorted(set(golds + preds))
    cm = confusion_matrix(golds, preds, labels=labels)

    return {
        "accuracy":       acc,
        "report":         report,
        "class_f1":       class_f1,
        "worst_intents":  worst,
        "confusion_matrix": cm.tolist(),
        "cm_labels":      labels,
    }


# ── 3. Escalation metrics (asymmetric) ─────────────────────────────────────
def escalation_metrics(results: list[dict]) -> dict:
    """
    False-auto-handle: gold=escalate but pred=auto (dangerous — missed escalation)
    False-escalate:    gold=auto but pred=escalate (costly — unnecessary routing)
    """
    tp = fp = fn = tn = 0  # escalate as positive class
    false_auto_cases = []
    false_escalate_cases = []

    for r in results:
        g = r["gold_decision"]
        p = r["pred_decision"]
        if g == "escalate" and p == "escalate": tp += 1
        elif g == "auto"    and p == "escalate":
            fp += 1
            false_escalate_cases.append(r)
        elif g == "escalate" and p == "auto":
            fn += 1
            false_auto_cases.append(r)
        elif g == "auto" and p == "auto": tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "escalation_precision":      precision,
        "escalation_recall":         recall,
        "escalation_f1":             f1,
        "false_auto_count":          fn,   # missed escalations (HIGH RISK)
        "false_escalate_count":      fp,   # unnecessary escalations (costly)
        "false_auto_examples":       false_auto_cases[:3],
        "false_escalate_examples":   false_escalate_cases[:3],
    }


# ── 4. Groundedness metric ─────────────────────────────────────────────────
def groundedness_metric(results: list[dict]) -> dict:
    """
    ROUGE-1 recall: what fraction of unigrams in the draft reply
    appear in the retrieved passages (from twcs.csv).
    Score = 0 means draft shares no words with retrieved passages.
    Score = 1 means every word in the draft is grounded in retrievals.
    This is a proxy — not perfect, but reproducible and data-only.
    """
    import re

    def tokens(text: str) -> set:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return set(text.split()) - {
            "i", "a", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "is", "are", "was", "were", "have", "has",
            "you", "your", "we", "our", "it", "its", "be", "can",
            "will", "would", "please", "thank", "thanks", "hi", "hello",
        }

    scores = []
    for r in results:
        if not r["draft_reply"] or not r["retrieved_hits"]:
            continue
        draft_tokens = tokens(r["draft_reply"])
        retrieved_text = " ".join(
            h["reply_text"] for h in r["retrieved_hits"] if isinstance(h, dict)
        )
        retrieved_tokens = tokens(retrieved_text)
        if not draft_tokens:
            continue
        overlap = draft_tokens & retrieved_tokens
        rouge1_recall = len(overlap) / len(draft_tokens)
        scores.append(rouge1_recall)

    mean_g = sum(scores) / len(scores) if scores else 0.0
    return {
        "mean_groundedness":   mean_g,
        "n_scored":            len(scores),
        "groundedness_scores": scores,
    }


# ── 5. Format report ───────────────────────────────────────────────────────
def format_report(intent_m, esc_m, ground_m, results) -> str:
    lines = [
        "=" * 65,
        "EVALUATION METRICS REPORT",
        "Source: golden_set.csv (from twcs.csv only)",
        "=" * 65,
        "",
        f"[INTENT CLASSIFICATION]",
        f"  Accuracy:           {intent_m['accuracy']:.3f}",
        f"  Macro F1:           {intent_m['report']['macro avg']['f1-score']:.3f}",
        f"  Weighted F1:        {intent_m['report']['weighted avg']['f1-score']:.3f}",
        "",
        "  Per-class F1:",
    ]
    for intent, f1 in sorted(intent_m["class_f1"].items(), key=lambda x: -x[1]):
        lines.append(f"    {intent:<35} F1={f1:.3f}")
    lines += [
        "",
        f"  Worst 3 intents by F1:",
    ]
    for intent, f1 in intent_m["worst_intents"]:
        lines.append(f"    {intent:<35} F1={f1:.3f}")

    lines += [
        "",
        "[ESCALATION (escalate = positive class)]",
        f"  Precision:          {esc_m['escalation_precision']:.3f}",
        f"  Recall:             {esc_m['escalation_recall']:.3f}",
        f"  F1:                 {esc_m['escalation_f1']:.3f}",
        f"  TP={esc_m['tp']}  FP={esc_m['fp']}  FN={esc_m['fn']}  TN={esc_m['tn']}",
        f"  FALSE-AUTO-HANDLE (missed escalations):  {esc_m['false_auto_count']}  ← HIGH RISK",
        f"  FALSE-ESCALATE (unnecessary routing):    {esc_m['false_escalate_count']}",
        "",
        "  False-auto examples (gold=escalate, pred=auto):",
    ]
    for ex in esc_m["false_auto_examples"]:
        lines.append(f"    [{ex['message_id']}] {ex['input_text'][:80]}")

    lines += [
        "",
        "[REPLY GROUNDEDNESS]",
        f"  Mean ROUGE-1 recall vs retrieved passages: {ground_m['mean_groundedness']:.3f}",
        f"  (1.0 = every draft word is in a retrieved AmazonHelp reply)",
        f"  (0.0 = draft shares no words with retrievals)",
        f"  N scored: {ground_m['n_scored']}",
        "",
        "NOTE: Groundedness proxy only. See llm_judge.py for rubric-based quality.",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load golden set
    if not GOLDEN_CSV.exists():
        sys.exit(f"ERROR: {GOLDEN_CSV} not found. Run eval/build_golden_set.py first.")

    with open(GOLDEN_CSV, encoding="utf-8") as f:
        golden_rows = list(csv.DictReader(f))
    print(f"[INFO] Loaded {len(golden_rows)} golden examples")

    # Filter out unlabelled rows
    labelled = [r for r in golden_rows if r.get("labelled_by","") not in ("NEEDS_HUMAN_REVIEW","")]
    if len(labelled) < 10:
        print(f"[WARN] Only {len(labelled)} rows have been human-labelled.")
        print("       Running on ALL rows (treating seed labels as gold).")
        labelled = golden_rows

    # REDUCE TO REAL STRATIFIED SUBSET (Step 1 of fix)
    subset = []
    counts = collections.defaultdict(int)
    for r in labelled:
        intent = r["gold_intent"]
        if counts[intent] < 5:
            subset.append(r)
            counts[intent] += 1
    labelled = subset
    print(f"[INFO] Reduced to stratified subset of {len(labelled)} examples.")

    # Run agent (or load cached results)
    if RESULTS_JSON.exists():
        print(f"[INFO] Loading cached agent results from {RESULTS_JSON}")
        with open(RESULTS_JSON, encoding="utf-8") as f:
            results = json.load(f)
    else:
        # Check API key before attempting agent run
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY"))
        if not has_key:
            print("[WARN] No LLM API key found and no cached results exist.")
            print("       Add OPENAI_API_KEY or GEMINI_API_KEY to .env, then re-run.")
            print("       Baselines are already in eval/baselines_report.txt")
            sys.exit(0)
        print("[INFO] Running agent on golden set ...")
        t0 = time.time()
        results = run_agent_on_golden(labelled)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Agent run complete in {time.time()-t0:.1f}s. Cached to {RESULTS_JSON}")

    # Compute metrics
    print("[INFO] Computing metrics ...")
    intent_m = intent_metrics(results)
    esc_m    = escalation_metrics(results)
    ground_m = groundedness_metric(results)

    report_str = format_report(intent_m, esc_m, ground_m, results)
    print("\n" + report_str.encode("cp1252", errors="replace").decode("cp1252"))

    REPORT_TXT.write_text(report_str, encoding="utf-8")
    print(f"\n[INFO] Report saved -> {REPORT_TXT}")
