"""
Baseline 3 — Retrieval-only agent (no LLM).

Intent:    keyword seed classifier (same as Baseline 2)
Decision:  taxonomy escalation defaults
Draft:     verbatim top-1 retrieved AmazonHelp reply from twcs.csv

Purpose:
  - Runs the full eval pipeline with ZERO API calls
  - Produces agent_results.json so metrics.py/llm_judge.py can run
  - Acts as a third baseline: how well does pure retrieval do?
  - Groundedness will be 1.0 by construction (draft = retrieved reply)
    so it sets the ceiling for that metric

IMPORTANT: Label in metrics report is "retrieval_only_baseline", not "agent".
           The LLM agent results will overwrite this file once an API key is set.
"""

import csv
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_CSV   = ROOT / "eval" / "golden_set.csv"
RESULTS_JSON = ROOT / "eval" / "agent_results.json"

# Intent seeds (same as taxonomy.py / baselines.py)
INTENT_SEEDS = {
    "ORDER_DELIVERY_STATUS":     ["where","order","package","tracking","delivery","delivered","arrive","arrived","ship","shipping","shipment","status","track","dispatched","carrier","late","delayed","delay","not received","missing","address","wrong address"],
    "RETURN_REFUND_REPLACEMENT": ["return","refund","replacement","exchange","money back","reimburse","damaged","broken","defective","wrong item","incorrect","not as described","replace"],
    "ACCOUNT_ACCESS":            ["account","login","sign in","password","locked","access","verify","verification","suspended","banned","hacked","unauthorized","forgot password","reset","cannot log"],
    "CHARGE_PAYMENT_BILLING":    ["charge","charged","billing","bill","payment","paid","double charged","overcharged","invoice","receipt","card","credit card","debit","fee","cost","gift card"],
    "PRIME_SUBSCRIPTION":        ["prime","prime membership","prime video","prime subscription","free trial","membership","annual","student prime","cancel prime","prime benefits"],
    "PRODUCT_COMPLAINT":         ["quality","fake","counterfeit","not working","broken","poor quality","disappointed","terrible","awful","complaint","defective","used","opened","expired","not genuine","seller","third party","marketplace","vendor"],
    "ORDER_CANCELLATION":        ["cancel","cancelled","cancellation","cancel order","stop order","don't want","do not want"],
    "DEVICE_APP_TECHNICAL":      ["kindle","echo","alexa","fire tv","firestick","app","website","site","crash","crashing","not loading","streaming","buffering","technical","software","update","bug"],
}

ESCALATION_DEFAULTS = {
    "ORDER_DELIVERY_STATUS":     ("auto",     "Standard tracking inquiry — no sensitive data required."),
    "RETURN_REFUND_REPLACEMENT": ("auto",     "Standard return policy — bot can provide return instructions."),
    "ACCOUNT_ACCESS":            ("escalate", "Account security requires identity verification by human."),
    "CHARGE_PAYMENT_BILLING":    ("escalate", "Billing dispute requires human review of payment records."),
    "PRIME_SUBSCRIPTION":        ("auto",     "Membership steps are publicly documented."),
    "PRODUCT_COMPLAINT":         ("auto",     "Standard refund/replacement offer applies."),
    "ORDER_CANCELLATION":        ("auto",     "Pre-shipment cancellation is automated."),
    "DEVICE_APP_TECHNICAL":      ("auto",     "Standard troubleshooting steps available."),
}

ESCALATION_TRIGGERS = [
    "lawyer","lawsuit","legal","police","fraud","identity theft",
    "stolen","threatening","unsafe","injury","hurt","hospital",
    "health risk","media","press","bbc","cnn",
]


def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_classify(text):
    t = clean(text)
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in INTENT_SEEDS.items()}
    best = max(scores, key=scores.get)
    return (best, round(min(scores[best] / 5.0, 1.0), 2)) if scores[best] > 0 else ("ORDER_DELIVERY_STATUS", 0.1)


def decide_escalation(intent, customer_text):
    text_lower = customer_text.lower()
    triggers = [kw for kw in ESCALATION_TRIGGERS if kw in text_lower]
    if triggers:
        return "escalate", f"High-risk keyword(s) detected: {triggers}"
    decision, reason = ESCALATION_DEFAULTS.get(intent, ("auto", "Default policy."))
    return decision, reason


if __name__ == "__main__":
    print("[INFO] Loading retriever ...")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from retrieval.retrieve import Retriever
    retriever = Retriever()

    with open(GOLDEN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labelled = [r for r in rows if r.get("labelled_by", "") not in ("NEEDS_HUMAN_REVIEW", "")]
    print(f"[INFO] Running retrieval-only baseline on {len(labelled)} labelled examples ...")

    results = []
    t0 = time.time()
    for i, row in enumerate(labelled):
        print(f"\r  [{i+1}/{len(labelled)}]", end="", flush=True)
        text = row["input_text"]

        # 1. retrieve
        hits = retriever.query(text, top_k=5)

        # 2. classify (keyword)
        intent, confidence = keyword_classify(text)

        # 3. escalate
        decision, reason = decide_escalation(intent, text)

        # 4. draft = top retrieved reply verbatim (no LLM)
        draft = hits[0]["reply_text"] if hits else "Please contact us for assistance with your request."

        results.append({
            "message_id":    row["message_id"],
            "gold_intent":   row["gold_intent"],
            "gold_decision": row["gold_decision"],
            "pred_intent":   intent,
            "pred_decision": decision,
            "pred_reason":   reason,
            "draft_reply":   draft,
            "retrieved_hits": hits,
            "confidence":    confidence,
            "input_text":    text,
            "reference_note": row.get("reference_resolution_note", ""),
            "agent_type":    "retrieval_only_baseline",
        })

    print(f"\n[INFO] Done in {time.time()-t0:.1f}s")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(results)} results -> {RESULTS_JSON}")
    print("[INFO] Now run: python eval/metrics.py")
    print("[NOTE] draft_reply = verbatim retrieved reply; groundedness will be ~1.0 by construction.")
    print("[NOTE] Overwrite with LLM agent results by running: python run_pipeline.py (with API key)")
