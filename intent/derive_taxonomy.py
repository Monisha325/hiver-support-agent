"""
Phase 1 — Intent taxonomy derivation for AmazonHelp.
Source: twcs.csv (Kaggle Customer Support on Twitter) — ONLY.
Banking77 is NOT loaded or referenced here; it was consulted only as a
shape reference (how many intents, how granular) in the design notes.

This script:
  1. Extracts all AmazonHelp conversation threads from twcs.csv
  2. Subsamples to CORPUS_N inbound customer messages for analysis
  3. Runs TF-IDF + top-keyword analysis per manual cluster seed
  4. Prints candidate intents with supporting real examples from the data
  5. Writes intent/taxonomy_candidates.txt for human review

Output files:
  intent/amazon_threads.csv       — all AmazonHelp inbound+outbound rows
  intent/taxonomy_candidates.txt  — candidate intents with examples
"""

import pathlib, re, sys, collections
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── paths ─────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parent.parent
RAW_CSV    = ROOT / "data" / "raw" / "twcs.csv"
INTENT_DIR = ROOT / "intent"
INTENT_DIR.mkdir(exist_ok=True)

BRAND_ID   = "AmazonHelp"
CORPUS_N   = 50_000   # subsample for working corpus (no full-dataset assumption)
SEED       = 42

# ── 1. Load & extract AmazonHelp threads ─────────────────────────────────
print("[INFO] Loading twcs.csv ...")
# Read full file — we need all rows to reconstruct conversation chains
# (the brand replies reference inbound tweet IDs from anywhere in the file)
df = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
print(f"[INFO] Total rows: {len(df):,}")

df["inbound"] = df["inbound"].str.strip().str.lower().map(
    {"true": True, "false": False}
)

# Brand outbound replies
brand_out = df[(df["author_id"] == BRAND_ID) & (df["inbound"] == False)].copy()
print(f"[INFO] AmazonHelp outbound replies: {len(brand_out):,}")

# IDs of tweets that AmazonHelp replied to
replied_ids = set(brand_out["in_response_to_tweet_id"].dropna().unique())
print(f"[INFO] Unique inbound tweets AmazonHelp replied to: {len(replied_ids):,}")

# The inbound (customer) messages AmazonHelp actually responded to
inbound = df[(df["inbound"] == True) & (df["tweet_id"].isin(replied_ids))].copy()
print(f"[INFO] Matched inbound customer tweets: {len(inbound):,}")

# Save the full thread corpus (inbound + outbound pairs)
brand_out_indexed  = brand_out.set_index("in_response_to_tweet_id")[
    ["tweet_id", "text", "created_at"]
].rename(columns={"tweet_id": "reply_id", "text": "reply_text", "created_at": "reply_at"})

threads = inbound.join(brand_out_indexed, on="tweet_id", how="left")
threads.to_csv(INTENT_DIR / "amazon_threads.csv", index=False)
print(f"[INFO] Saved {len(threads):,} threads -> intent/amazon_threads.csv")

# ── 2. Subsample for taxonomy derivation ─────────────────────────────────
# Stratified by nothing yet (we have no labels) — random sample
sample = inbound.sample(n=min(CORPUS_N, len(inbound)), random_state=SEED).copy()
print(f"[INFO] Working corpus: {len(sample):,} inbound tweets")

# ── 3. Clean text ─────────────────────────────────────────────────────────
def clean(text: str) -> str:
    """Minimal cleaning: lowercase, strip URLs/mentions/punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)             # @mentions
    text = re.sub(r"#\w+", "", text)             # hashtags
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

sample["clean_text"] = sample["text"].apply(clean)
corpus = sample["clean_text"].tolist()

# ── 4. Global TF-IDF — top 200 terms across entire corpus ─────────────────
print("[INFO] Running TF-IDF on corpus ...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5,
    max_df=0.6,
)
X = tfidf.fit_transform(corpus)
feature_names = tfidf.get_feature_names_out()
mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
top_idx = mean_tfidf.argsort()[::-1][:200]
top_terms = [(feature_names[i], round(float(mean_tfidf[i]), 5)) for i in top_idx]

print("\n[TOP 50 TF-IDF TERMS ACROSS ALL AMAZON INBOUND TWEETS]")
for term, score in top_terms[:50]:
    print(f"  {term:<30} {score:.5f}")

# ── 5. Keyword-based intent seeding ───────────────────────────────────────
# These seed keyword lists were built by reading the top TF-IDF terms above
# and grouping by apparent topic — NOT from Banking77 or any external source.
# The actual groupings were derived iteratively from the data.

INTENT_SEEDS = {
    "ORDER_DELIVERY_STATUS": [
        "where", "order", "package", "tracking", "delivery", "delivered",
        "arrive", "arrived", "expected", "ship", "shipping", "shipment",
        "status", "track", "dispatched", "carrier", "usps", "ups", "fedex",
        "estimated", "arrival", "late", "delayed", "delay", "not received",
        "missing package",
    ],
    "RETURN_REFUND_REPLACEMENT": [
        "return", "refund", "replacement", "exchange", "money back",
        "reimburse", "reimbursement", "credit", "sent back", "sending back",
        "damaged", "broken", "defective", "wrong item", "incorrect",
        "not as described", "replace", "send another",
    ],
    "ACCOUNT_LOGIN_ACCESS": [
        "account", "login", "sign in", "password", "locked", "access",
        "verify", "verification", "email", "2fa", "two factor",
        "suspended", "banned", "hacked", "unauthorized", "forgot password",
        "reset", "cannot log", "can't log",
    ],
    "ORDER_CANCELLATION": [
        "cancel", "cancelled", "cancellation", "cancel order",
        "stop order", "don't want", "do not want", "withdraw",
    ],
    "CHARGE_PAYMENT_BILLING": [
        "charge", "charged", "billing", "bill", "payment", "paid",
        "double charged", "overcharged", "invoice", "receipt",
        "card", "credit card", "debit", "unauthorized charge",
        "prime charge", "subscription", "fee", "cost",
    ],
    "PRIME_SUBSCRIPTION": [
        "prime", "prime membership", "prime video", "prime day",
        "prime subscription", "free trial", "membership", "annual",
        "student prime", "cancel prime", "prime benefits",
    ],
    "PRODUCT_QUALITY_COMPLAINT": [
        "quality", "fake", "counterfeit", "not working", "doesn't work",
        "broken", "poor quality", "disappointed", "terrible", "awful",
        "complaint", "issue with", "problem with", "defective product",
        "used", "opened", "expired", "wrong", "not genuine",
    ],
    "SELLER_THIRD_PARTY": [
        "seller", "third party", "marketplace", "vendor", "sold by",
        "fulfilled", "independent seller", "merchant", "fraudulent seller",
        "scam seller",
    ],
    "DELIVERY_ADDRESS_CHANGE": [
        "address", "change address", "wrong address", "update address",
        "redirect", "delivery address", "shipping address",
    ],
    "GENERAL_INQUIRY_OTHER": [
        "help", "question", "wondering", "how do", "can you",
        "what is", "please", "information", "assist",
    ],
}

# ── 6. Score each message against each intent seed ────────────────────────
print("\n[INFO] Scoring messages against intent seeds ...")

def seed_score(text: str, keywords: list) -> int:
    """Count how many seed keywords appear in the text."""
    return sum(1 for kw in keywords if kw in text)

for intent, kws in INTENT_SEEDS.items():
    sample[f"score_{intent}"] = sample["clean_text"].apply(
        lambda t: seed_score(t, kws)
    )

score_cols = [f"score_{k}" for k in INTENT_SEEDS]
sample["best_intent"] = sample[score_cols].idxmax(axis=1).str.replace("score_", "")
sample["best_score"]  = sample[score_cols].max(axis=1)

# Treat score=0 as unclassified
sample.loc[sample["best_score"] == 0, "best_intent"] = "UNCLASSIFIED"

intent_counts = sample["best_intent"].value_counts()
print("\n[INTENT DISTRIBUTION — keyword-seeded, from actual AmazonHelp inbound tweets]")
total = len(sample)
for intent, count in intent_counts.items():
    print(f"  {intent:<35} {count:>6,}  ({100*count/total:5.1f}%)")

# ── 7. Pull 3 real examples per intent ────────────────────────────────────
print("\n[REAL EXAMPLES PER INTENT — from AmazonHelp inbound tweets in twcs.csv]")
examples = {}
for intent in INTENT_SEEDS:
    subset = sample[sample["best_intent"] == intent].nlargest(3, f"score_{intent}")
    examples[intent] = subset["text"].tolist()

for intent, exs in examples.items():
    print(f"\n--- {intent} ---")
    for i, ex in enumerate(exs, 1):
        print(f"  [{i}] {ex[:200]}")

# ── 8. Top unigrams per intent (from messages assigned to that intent) ────
print("\n[TOP UNIGRAMS PER INTENT]")
for intent in INTENT_SEEDS:
    subset_texts = sample[sample["best_intent"] == intent]["clean_text"].tolist()
    if len(subset_texts) < 5:
        continue
    cv = TfidfVectorizer(max_features=200, ngram_range=(1,1),
                         stop_words="english", min_df=2)
    try:
        Xs = cv.fit_transform(subset_texts)
        terms = cv.get_feature_names_out()
        scores = np.asarray(Xs.mean(axis=0)).flatten()
        top = sorted(zip(terms, scores), key=lambda x: -x[1])[:10]
        print(f"  {intent}: {[t for t,_ in top]}")
    except Exception:
        pass

# ── 9. Write taxonomy candidate report ────────────────────────────────────
out_lines = [
    "=" * 70,
    "PHASE 1 — INTENT TAXONOMY CANDIDATES",
    "Brand: AmazonHelp | Source: twcs.csv ONLY",
    "Banking77 referenced only for shape (count/granularity), NOT for labels",
    "=" * 70,
    "",
    f"Working corpus: {len(sample):,} inbound AmazonHelp tweets",
    "",
    "CANDIDATE INTENTS (frequency + real examples from data):",
    "-" * 70,
]

for intent, count in intent_counts.items():
    if intent == "UNCLASSIFIED":
        continue
    pct = 100 * count / total
    out_lines += [
        f"",
        f"INTENT: {intent}",
        f"  Count: {count:,}  ({pct:.1f}% of corpus)",
        f"  Real examples (from twcs.csv):",
    ]
    for ex in examples.get(intent, []):
        out_lines.append(f"    - {ex[:200]}")

unclassified = intent_counts.get("UNCLASSIFIED", 0)
out_lines += [
    "",
    f"UNCLASSIFIED (score=0 on all seeds): {unclassified:,} "
    f"({100*unclassified/total:.1f}%)",
    "",
    "NOTE: 'UNCLASSIFIED' candidates will be inspected to either fold into",
    "  existing intents or promote to new intents before finalising taxonomy.",
]

out_path = INTENT_DIR / "taxonomy_candidates.txt"
out_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"\n[INFO] Taxonomy candidates written -> {out_path}")
print("[INFO] Phase 1 derivation complete. Review taxonomy_candidates.txt.")
