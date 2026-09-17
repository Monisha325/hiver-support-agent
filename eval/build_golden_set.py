"""
Phase 5 — Golden evaluation set builder.

Produces eval/golden_set.csv with 150-250 hand-labelled examples.

Schema (per assignment):
  message_id, input_text, gold_intent, gold_decision,
  gold_decision_reason, reference_resolution_note,
  sampling_batch, labelled_by

Sampling strategy (to satisfy deliberate coverage requirement):
  - Batch A (80 examples): proportional to intent frequency in corpus
  - Batch B (60 examples): oversampled from tail intents (DEVICE_APP_TECHNICAL,
    ORDER_CANCELLATION, PRODUCT_COMPLAINT, SELLER tail)
  - Batch C (40 examples): deliberately hard cases (short messages,
    ambiguous intent, multi-intent, non-English-but-in-corpus)
  - Batch D (30 examples): escalation-positive cases
  - Batch E (20 examples): cases where auto-handle is clear-cut
  Total target: 230 examples (within 150-250 range)

This script:
  1. Samples from the AmazonHelp inbound corpus (twcs.csv derived threads)
  2. Writes a CSV pre-populated with:
       - input_text (from data)
       - gold_intent (pre-filled by keyword seed — MUST be human-verified)
       - gold_decision (pre-filled by taxonomy default — MUST be human-verified)
  3. Prints a labelling guide so you can open the CSV and correct each row

IMPORTANT: The pre-filled labels are STARTING POINTS only.
Every row must be read and corrected by a human before use in evaluation.
The script prints the first 10 rows for review.
"""

import json
import pathlib
import re
import random
import csv
import sys

import pandas as pd

ROOT        = pathlib.Path(__file__).parent.parent
RAW_CSV     = ROOT / "data" / "raw" / "twcs.csv"
CORPUS_PATH = ROOT / "retrieval" / "corpus.jsonl"
OUT_PATH    = ROOT / "eval" / "golden_set.csv"
ROOT / "eval"

BRAND_ID = "AmazonHelp"
SEED     = 2024

INTENT_SEEDS = {
    "ORDER_DELIVERY_STATUS":       ["where","order","package","tracking","delivery","delivered","arrive","arrived","ship","shipping","shipment","status","track","dispatched","carrier","late","delayed","delay","not received","missing","address","wrong address"],
    "RETURN_REFUND_REPLACEMENT":   ["return","refund","replacement","exchange","money back","reimburse","reimbursement","damaged","broken","defective","wrong item","incorrect","not as described","replace"],
    "ACCOUNT_ACCESS":              ["account","login","sign in","password","locked","access","verify","verification","suspended","banned","hacked","unauthorized","forgot password","reset","cannot log"],
    "CHARGE_PAYMENT_BILLING":      ["charge","charged","billing","bill","payment","paid","double charged","overcharged","invoice","receipt","card","credit card","debit","unauthorized charge","fee","cost","gift card"],
    "PRIME_SUBSCRIPTION":          ["prime","prime membership","prime video","prime subscription","free trial","membership","annual","student prime","cancel prime","prime benefits"],
    "PRODUCT_COMPLAINT":           ["quality","fake","counterfeit","not working","broken","poor quality","disappointed","terrible","awful","complaint","defective","used","opened","expired","not genuine","seller","third party","marketplace","vendor","fulfilled"],
    "ORDER_CANCELLATION":          ["cancel","cancelled","cancellation","cancel order","stop order","don't want","do not want"],
    "DEVICE_APP_TECHNICAL":        ["kindle","echo","alexa","fire tv","firestick","app","website","site","crash","crashing","not loading","streaming","buffering","prime video","technical","software","update","bug"],
}

ESCALATION_DEFAULTS = {
    "ORDER_DELIVERY_STATUS":       "auto",
    "RETURN_REFUND_REPLACEMENT":   "auto",
    "ACCOUNT_ACCESS":              "escalate",
    "CHARGE_PAYMENT_BILLING":      "escalate",
    "PRIME_SUBSCRIPTION":          "auto",
    "PRODUCT_COMPLAINT":           "auto",
    "ORDER_CANCELLATION":          "auto",
    "DEVICE_APP_TECHNICAL":        "auto",
}

ESCALATION_REASONS = {
    "ORDER_DELIVERY_STATUS":       "Standard tracking update; no sensitive data required.",
    "RETURN_REFUND_REPLACEMENT":   "Standard return policy; bot can provide return label instructions.",
    "ACCOUNT_ACCESS":              "Account security issue requires identity verification by human agent.",
    "CHARGE_PAYMENT_BILLING":      "Billing dispute requires human review of payment records.",
    "PRIME_SUBSCRIPTION":          "Membership information is publicly documented; bot can explain steps.",
    "PRODUCT_COMPLAINT":           "Standard refund/replacement offer; bot can initiate process.",
    "ORDER_CANCELLATION":          "Pre-shipment cancellation is automated; no human required.",
    "DEVICE_APP_TECHNICAL":        "Standard troubleshooting steps; bot can guide customer.",
}


def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def seed_intent(text):
    scores = {k: sum(1 for kw in kws if kw in text) for k, kws in INTENT_SEEDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNCLASSIFIED"


def is_english(text):
    """Rough heuristic: >60% ASCII printable chars."""
    if not text: return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.6


print("[INFO] Loading corpus from retrieval/corpus.jsonl ...")
if not CORPUS_PATH.exists():
    sys.exit(f"ERROR: {CORPUS_PATH} not found. Run retrieval/build_index.py first.")

records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        rec["clean"] = clean(rec["customer_text"])
        rec["intent_seed"] = seed_intent(rec["clean"])
        records.append(rec)

df = pd.DataFrame(records)
df = df[df["customer_text"].apply(is_english)].copy()
print(f"[INFO] English-only corpus: {len(df):,} rows")

random.seed(SEED)
pd.options.mode.chained_assignment = None

rows = []
msg_id = 1

# ── Batch A: proportional sample (80 examples) ────────────────────────────
print("[INFO] Batch A: proportional sample ...")
intent_freq = df["intent_seed"].value_counts(normalize=True)
for intent, pct in intent_freq.items():
    if intent == "UNCLASSIFIED":
        continue
    n = max(1, round(pct * 80))
    subset = df[df["intent_seed"] == intent].sample(n=min(n, len(df[df["intent_seed"]==intent])), random_state=SEED)
    for _, row in subset.iterrows():
        rows.append({
            "message_id":                f"MSG_{msg_id:04d}",
            "input_text":                row["customer_text"],
            "gold_intent":               intent,
            "gold_decision":             ESCALATION_DEFAULTS.get(intent, "auto"),
            "gold_decision_reason":      ESCALATION_REASONS.get(intent, ""),
            "reference_resolution_note": row["reply_text"][:300],
            "sampling_batch":            "A_proportional",
            "labelled_by":               "NEEDS_HUMAN_REVIEW",
        })
        msg_id += 1

# ── Batch B: tail intents oversampled (60 examples) ───────────────────────
print("[INFO] Batch B: tail intent oversample ...")
TAIL_INTENTS = ["DEVICE_APP_TECHNICAL", "ORDER_CANCELLATION",
                "PRODUCT_COMPLAINT", "CHARGE_PAYMENT_BILLING"]
for intent in TAIL_INTENTS:
    subset = df[df["intent_seed"] == intent]
    n = min(15, len(subset))
    if n == 0: continue
    for _, row in subset.sample(n=n, random_state=SEED+1).iterrows():
        rows.append({
            "message_id":                f"MSG_{msg_id:04d}",
            "input_text":                row["customer_text"],
            "gold_intent":               intent,
            "gold_decision":             ESCALATION_DEFAULTS.get(intent, "auto"),
            "gold_decision_reason":      ESCALATION_REASONS.get(intent, ""),
            "reference_resolution_note": row["reply_text"][:300],
            "sampling_batch":            "B_tail_oversample",
            "labelled_by":               "NEEDS_HUMAN_REVIEW",
        })
        msg_id += 1

# ── Batch C: hard cases — short messages, ambiguous (40 examples) ─────────
print("[INFO] Batch C: hard/ambiguous cases ...")
# Short messages (harder to classify)
short = df[df["customer_text"].str.len().between(20, 80)].sample(
    n=min(20, len(df[df["customer_text"].str.len().between(20,80)])),
    random_state=SEED+2
)
# Unclassified (genuinely ambiguous)
unclassified = df[df["intent_seed"] == "UNCLASSIFIED"].sample(
    n=min(20, len(df[df["intent_seed"]=="UNCLASSIFIED"])), random_state=SEED+3
)
for subset, batch_tag in [(short, "C_short"), (unclassified, "C_ambiguous")]:
    for _, row in subset.iterrows():
        rows.append({
            "message_id":                f"MSG_{msg_id:04d}",
            "input_text":                row["customer_text"],
            "gold_intent":               row["intent_seed"],   # likely UNCLASSIFIED or uncertain
            "gold_decision":             ESCALATION_DEFAULTS.get(row["intent_seed"], "auto"),
            "gold_decision_reason":      "AMBIGUOUS — human labeller must decide",
            "reference_resolution_note": row["reply_text"][:300],
            "sampling_batch":            batch_tag,
            "labelled_by":               "NEEDS_HUMAN_REVIEW",
        })
        msg_id += 1

# ── Batch D: escalation-positive (30 examples) ────────────────────────────
print("[INFO] Batch D: escalation-positive ...")
ESCALATION_INTENTS = ["ACCOUNT_ACCESS", "CHARGE_PAYMENT_BILLING"]
for intent in ESCALATION_INTENTS:
    subset = df[df["intent_seed"] == intent]
    n = min(15, len(subset))
    if n == 0: continue
    for _, row in subset.sample(n=n, random_state=SEED+4).iterrows():
        rows.append({
            "message_id":                f"MSG_{msg_id:04d}",
            "input_text":                row["customer_text"],
            "gold_intent":               intent,
            "gold_decision":             "escalate",
            "gold_decision_reason":      ESCALATION_REASONS[intent],
            "reference_resolution_note": row["reply_text"][:300],
            "sampling_batch":            "D_escalation_positive",
            "labelled_by":               "NEEDS_HUMAN_REVIEW",
        })
        msg_id += 1

# ── Batch E: clear auto-handle (20 examples) ──────────────────────────────
print("[INFO] Batch E: clear auto-handle ...")
AUTO_INTENTS = ["ORDER_DELIVERY_STATUS", "PRIME_SUBSCRIPTION"]
for intent in AUTO_INTENTS:
    subset = df[df["intent_seed"] == intent]
    n = min(10, len(subset))
    for _, row in subset.sample(n=n, random_state=SEED+5).iterrows():
        rows.append({
            "message_id":                f"MSG_{msg_id:04d}",
            "input_text":                row["customer_text"],
            "gold_intent":               intent,
            "gold_decision":             "auto",
            "gold_decision_reason":      ESCALATION_REASONS[intent],
            "reference_resolution_note": row["reply_text"][:300],
            "sampling_batch":            "E_auto_handle",
            "labelled_by":               "NEEDS_HUMAN_REVIEW",
        })
        msg_id += 1

# ── Deduplicate & cap at 250 ──────────────────────────────────────────────
seen_texts = set()
deduped = []
for r in rows:
    t = r["input_text"][:100]   # fuzzy dedup by first 100 chars
    if t not in seen_texts:
        seen_texts.add(t)
        deduped.append(r)

deduped = deduped[:250]
print(f"\n[INFO] Total golden set rows: {len(deduped)} (target 150-250)")

# ── Write CSV ─────────────────────────────────────────────────────────────
(ROOT / "eval").mkdir(exist_ok=True)
fieldnames = [
    "message_id", "input_text", "gold_intent", "gold_decision",
    "gold_decision_reason", "reference_resolution_note",
    "sampling_batch", "labelled_by",
]
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)

print(f"[INFO] Golden set saved -> {OUT_PATH}")
print("\n" + "="*60)
print("LABELLING GUIDE")
print("="*60)
print("Open eval/golden_set.csv and review EVERY row:")
print("  gold_intent       — correct if the seed keyword got it wrong")
print("  gold_decision     — correct based on full message context")
print("  gold_decision_reason — update to reflect your actual reasoning")
print("  labelled_by       — replace 'NEEDS_HUMAN_REVIEW' with your name")
print("\nBatch C (ambiguous) rows MUST be manually resolved.")
print("UNCLASSIFIED rows must be assigned to one of the 8 intents or marked OOS.")
print("\nFirst 10 rows for review:")
for r in deduped[:10]:
    print(f"\n  [{r['message_id']}] [{r['sampling_batch']}]")
    print(f"  TEXT:     {r['input_text'][:120]}")
    print(f"  INTENT:   {r['gold_intent']}  |  DECISION: {r['gold_decision']}")

batch_counts = pd.DataFrame(deduped)["sampling_batch"].value_counts()
print("\n\nBatch distribution:")
print(batch_counts.to_string())
