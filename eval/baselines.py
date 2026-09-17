"""
Phase 6 — Baselines for comparison.

Two baselines built from the SAME locked sources (twcs.csv only):

  Baseline 1 (TRIVIAL): Majority-class classifier — always predicts
    ORDER_DELIVERY_STATUS (the most frequent intent at ~32%).
    Escalation: always "auto".

  Baseline 2 (SIMPLE): Keyword/TF-IDF classifier — scores each message
    against the intent seed keyword lists (same seeds as taxonomy.py).
    No LLM, no embeddings. Escalation: taxonomy default.

Both are evaluated on the same golden_set.csv.
Results written to eval/baselines_report.txt.

Usage:
    python eval/baselines.py
"""

import csv, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_CSV   = ROOT / "eval" / "golden_set.csv"
REPORT_TXT   = ROOT / "eval" / "baselines_report.txt"

# ── Intent seeds (from taxonomy.py) ───────────────────────────────────────
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
    "ORDER_DELIVERY_STATUS":     "auto",
    "RETURN_REFUND_REPLACEMENT": "auto",
    "ACCOUNT_ACCESS":            "escalate",
    "CHARGE_PAYMENT_BILLING":    "escalate",
    "PRIME_SUBSCRIPTION":        "auto",
    "PRODUCT_COMPLAINT":         "auto",
    "ORDER_CANCELLATION":        "auto",
    "DEVICE_APP_TECHNICAL":      "auto",
}

MAJORITY_INTENT = "ORDER_DELIVERY_STATUS"


def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+",   "", text)
    text = re.sub(r"#\w+",   "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_classify(text):
    c = clean(text)
    scores = {k: sum(1 for kw in kws if kw in c) for k, kws in INTENT_SEEDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else MAJORITY_INTENT   # fallback to majority


def compute_metrics(golds_i, preds_i, golds_d, preds_d, name):
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # Filter out OOS/UNCLASSIFIED
    valid = [(g, p, gd, pd) for g, p, gd, pd in zip(golds_i, preds_i, golds_d, preds_d)
             if g not in ("UNCLASSIFIED", "OOS")]
    gi = [x[0] for x in valid]
    pi = [x[1] for x in valid]
    gd = [x[2] for x in valid]
    pd = [x[3] for x in valid]

    intent_acc   = accuracy_score(gi, pi)
    intent_macro = f1_score(gi, pi, average="macro", zero_division=0)
    intent_wt    = f1_score(gi, pi, average="weighted", zero_division=0)
    esc_acc      = accuracy_score(gd, pd)

    # Asymmetric escalation
    tp = sum(1 for g, p in zip(gd, pd) if g == "escalate" and p == "escalate")
    fp = sum(1 for g, p in zip(gd, pd) if g == "auto"     and p == "escalate")
    fn = sum(1 for g, p in zip(gd, pd) if g == "escalate" and p == "auto")
    tn = sum(1 for g, p in zip(gd, pd) if g == "auto"     and p == "auto")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "name":              name,
        "intent_accuracy":   intent_acc,
        "intent_macro_f1":   intent_macro,
        "intent_weighted_f1": intent_wt,
        "esc_precision":     prec,
        "esc_recall":        rec,
        "false_auto":        fn,
        "false_escalate":    fp,
        "n":                 len(gi),
    }


if __name__ == "__main__":
    if not GOLDEN_CSV.exists():
        sys.exit(f"ERROR: {GOLDEN_CSV} not found.")

    with open(GOLDEN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    golds_i = [r["gold_intent"]   for r in rows]
    golds_d = [r["gold_decision"]  for r in rows]
    texts   = [r["input_text"]     for r in rows]

    # ── Baseline 1: majority class ────────────────────────────────────────
    b1_preds_i = [MAJORITY_INTENT for _ in rows]
    b1_preds_d = ["auto"          for _ in rows]   # majority escalation = auto
    m1 = compute_metrics(golds_i, b1_preds_i, golds_d, b1_preds_d,
                         "Baseline 1 — Majority class (always ORDER_DELIVERY_STATUS / auto)")

    # ── Baseline 2: keyword TF-IDF ────────────────────────────────────────
    b2_preds_i = [keyword_classify(t) for t in texts]
    b2_preds_d = [ESCALATION_DEFAULTS.get(p, "auto") for p in b2_preds_i]
    m2 = compute_metrics(golds_i, b2_preds_i, golds_d, b2_preds_d,
                         "Baseline 2 — Keyword seed classifier (no LLM)")

    # ── Format report ─────────────────────────────────────────────────────
    lines = [
        "=" * 65,
        "BASELINES REPORT",
        "Source: golden_set.csv (derived from twcs.csv only)",
        "=" * 65, "",
    ]
    for m in [m1, m2]:
        lines += [
            f"--- {m['name']} ---",
            f"  N (excl. OOS):        {m['n']}",
            f"  Intent accuracy:      {m['intent_accuracy']:.3f}",
            f"  Intent macro F1:      {m['intent_macro_f1']:.3f}",
            f"  Intent weighted F1:   {m['intent_weighted_f1']:.3f}",
            f"  Escalation precision: {m['esc_precision']:.3f}",
            f"  Escalation recall:    {m['esc_recall']:.3f}",
            f"  False-auto-handle:    {m['false_auto']}   ← missed escalations",
            f"  False-escalate:       {m['false_escalate']}   ← unnecessary routing",
            "",
        ]

    out = "\n".join(lines)
    print(out)
    REPORT_TXT.write_text(out, encoding="utf-8")
    print(f"[INFO] Saved -> {REPORT_TXT}")

    # Also save as JSON for metrics.py comparison
    (ROOT / "eval" / "baselines_results.json").write_text(
        json.dumps([m1, m2], indent=2), encoding="utf-8"
    )
