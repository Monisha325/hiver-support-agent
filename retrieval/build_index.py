"""
Phase 2 — Build FAISS retrieval index over AmazonHelp's resolved threads.

Source: twcs.csv ONLY. No external data.

What this does:
  1. Loads all AmazonHelp inbound+outbound conversation pairs from twcs.csv
  2. Subsamples to CORPUS_N complete threads (customer msg + brand reply)
  3. Embeds customer messages with a lightweight sentence-transformer
  4. Builds a FAISS flat-L2 index
  5. Saves index + metadata so retrieval/retrieve.py can query it at runtime

Output files:
  retrieval/index.faiss       — FAISS index (customer-message embeddings)
  retrieval/corpus.jsonl      — metadata: {id, customer_text, reply_text, intent_seed}
  retrieval/build_log.txt     — build stats (row counts, timing)
"""

import json
import pathlib
import sys
import time
import re

import numpy as np
import pandas as pd

ROOT         = pathlib.Path(__file__).parent.parent
RAW_CSV      = ROOT / "data" / "raw" / "twcs.csv"
RETRIEVAL_DIR = ROOT / "retrieval"
RETRIEVAL_DIR.mkdir(exist_ok=True)

BRAND_ID   = "AmazonHelp"
CORPUS_N   = 40_000   # threads to index — fits in ~1 GB RAM for embeddings
SEED       = 42
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~80 MB, fast on CPU

# Intent seed keywords (mirrors taxonomy.py — kept inline to avoid circular import)
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


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def seed_intent(text: str) -> str:
    """Assign the highest-scoring seed intent (for metadata — not used in retrieval)."""
    scores = {k: sum(1 for kw in kws if kw in text) for k, kws in INTENT_SEEDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNCLASSIFIED"


# ── 1. Load & build thread pairs ─────────────────────────────────────────
t0 = time.time()
print("[INFO] Loading twcs.csv ...")
df = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
print(f"[INFO] Total rows: {len(df):,}")

df["inbound"] = df["inbound"].str.strip().str.lower().map({"true": True, "false": False})

brand_out = df[(df["author_id"] == BRAND_ID) & (df["inbound"] == False)].copy()
print(f"[INFO] AmazonHelp outbound replies: {len(brand_out):,}")

# Build lookup: inbound_tweet_id -> best reply text
# One inbound tweet can have multiple brand replies; keep the longest (most informative)
reply_lookup = (
    brand_out[brand_out["in_response_to_tweet_id"].notna()]
    .assign(reply_len=brand_out["text"].str.len())
    .sort_values("reply_len", ascending=False)
    .drop_duplicates(subset=["in_response_to_tweet_id"])
    .set_index("in_response_to_tweet_id")["text"]
    .to_dict()
)
print(f"[INFO] Unique inbound->reply mappings: {len(reply_lookup):,}")

# Get the inbound tweets that have a reply
inbound = df[
    (df["inbound"] == True) &
    (df["tweet_id"].isin(reply_lookup.keys()))
].copy()
print(f"[INFO] Inbound tweets with a reply: {len(inbound):,}")

# Attach reply
inbound["reply_text"] = inbound["tweet_id"].map(reply_lookup)

# Filter out pairs where either side is empty / very short
inbound = inbound[
    inbound["text"].str.len().fillna(0) > 20
].copy()
inbound = inbound[
    inbound["reply_text"].str.len().fillna(0) > 20
].copy()
print(f"[INFO] After length filter: {len(inbound):,} pairs")

# ── 2. Subsample ─────────────────────────────────────────────────────────
corpus = inbound.sample(n=min(CORPUS_N, len(inbound)), random_state=SEED).reset_index(drop=True)
print(f"[INFO] Subsampled to {len(corpus):,} threads")

corpus["clean_customer"] = corpus["text"].apply(clean)
corpus["intent_seed"]    = corpus["clean_customer"].apply(seed_intent)

intent_dist = corpus["intent_seed"].value_counts()
print("\n[INFO] Intent distribution in retrieval corpus:")
for intent, count in intent_dist.items():
    print(f"       {intent:<35} {count:>6,}  ({100*count/len(corpus):.1f}%)")

# ── 3. Embed customer messages ────────────────────────────────────────────
print(f"\n[INFO] Loading sentence-transformer: {MODEL_NAME} ...")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers")

model = SentenceTransformer(MODEL_NAME)
texts = corpus["clean_customer"].tolist()

print(f"[INFO] Embedding {len(texts):,} texts (batch_size=256) ...")
t_embed = time.time()
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,   # cosine via inner product
    convert_to_numpy=True,
)
print(f"[INFO] Embedding done in {time.time()-t_embed:.1f}s. Shape: {embeddings.shape}")

# ── 4. Build FAISS index ─────────────────────────────────────────────────
try:
    import faiss
except ImportError:
    sys.exit("Run: pip install faiss-cpu")

dim = embeddings.shape[1]
# IndexFlatIP: exact inner-product search (cosine, since embeddings are normalised)
# Simple and fully reproducible — no approximation.
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))
print(f"[INFO] FAISS index built. Vectors: {index.ntotal:,}, dim: {dim}")

index_path = RETRIEVAL_DIR / "index.faiss"
faiss.write_index(index, str(index_path))
print(f"[INFO] Index saved -> {index_path}")

# ── 5. Save corpus metadata ───────────────────────────────────────────────
corpus_path = RETRIEVAL_DIR / "corpus.jsonl"
with open(corpus_path, "w", encoding="utf-8") as f:
    for i, row in corpus.iterrows():
        record = {
            "idx":           int(i),
            "tweet_id":      str(row.get("tweet_id", "")),
            "customer_text": str(row.get("text", "")),
            "reply_text":    str(row.get("reply_text", "")),
            "intent_seed":   str(row.get("intent_seed", "")),
            "created_at":    str(row.get("created_at", "")),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"[INFO] Corpus metadata saved -> {corpus_path}")

# ── 6. Build log ─────────────────────────────────────────────────────────
elapsed = time.time() - t0
log = [
    "=" * 60,
    "PHASE 2 — RETRIEVAL INDEX BUILD LOG",
    "=" * 60,
    f"Source:           twcs.csv (Kaggle Customer Support on Twitter)",
    f"Brand:            AmazonHelp",
    f"Total CSV rows:   {len(df):,}",
    f"Brand replies:    {len(brand_out):,}",
    f"Matched pairs:    {len(inbound):,}",
    f"Indexed corpus:   {len(corpus):,}",
    f"Embedding model:  {MODEL_NAME}",
    f"Embedding dim:    {dim}",
    f"FAISS index type: IndexFlatIP (exact cosine)",
    f"Build time:       {elapsed:.1f}s",
    "",
    "Intent distribution:",
]
for intent, count in intent_dist.items():
    log.append(f"  {intent:<35} {count:>6,}  ({100*count/len(corpus):.1f}%)")

log_path = RETRIEVAL_DIR / "build_log.txt"
log_path.write_text("\n".join(log), encoding="utf-8")
print(f"\n[INFO] Build log -> {log_path}")
print(f"[INFO] Phase 2 complete in {elapsed:.1f}s")
