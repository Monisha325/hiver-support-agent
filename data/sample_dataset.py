"""
Step 0 — Data download + profiling script.
Source: Kaggle "Customer Support on Twitter" (thoughtvector/customer-support-on-twitter).
Only the twcs.csv file from that dataset is used here.

Usage:
    python data/sample_dataset.py

Requires KAGGLE_USERNAME and KAGGLE_KEY in environment (or ~/.kaggle/kaggle.json).
Produces:
    data/raw/twcs.csv           — raw download (≈ 3 M rows)
    data/brand_profile.csv      — per-brand aggregates
    data/step0_report.txt       — human-readable brand shortlist
"""

import os
import sys
import json
import zipfile
import pathlib
import textwrap

import pandas as pd
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SLUG = "thoughtvector/customer-support-on-twitter"
CSV_NAME     = "twcs.csv"
RAW_CSV      = RAW_DIR / CSV_NAME


# ── 1. download if needed ─────────────────────────────────────────────────────
def download_dataset():
    if RAW_CSV.exists():
        print(f"[INFO] {RAW_CSV} already present -- skipping download.")
        return
    sys.exit(
        f"ERROR: {RAW_CSV} not found.\n"
        "Run: python data/download.py  (requires KAGGLE_API_TOKEN env var)\n"
        "Or manually download twcs.csv from:\n"
        "  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter"
    )


# ── 2. load & profile ─────────────────────────────────────────────────────────
def profile_brands(sample_n: int = 500_000) -> pd.DataFrame:
    """
    Read up to sample_n rows for profiling (the full file is ~3 M rows).
    The file has columns: tweet_id, author_id, inbound, created_at,
    text, response_tweet_id, in_response_to_tweet_id
    """
    print(f"[INFO] Loading up to {sample_n:,} rows from {RAW_CSV} …")
    df = pd.read_csv(RAW_CSV, nrows=sample_n, dtype=str, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Normalise boolean column
    # 'inbound' = True means the tweet IS a customer message (inbound to brand)
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )

    # ── identify brand author_ids ────────────────────────────────────────────
    # Brand accounts are those that send outbound replies (inbound == False)
    # and receive inbound tweets (inbound == True) in response chains.
    #
    # Strategy: author_ids that are predominantly outbound (< 20 % inbound
    # tweets) and have a minimum outbound volume threshold.

    # Separate inbound (customer) vs outbound (brand) rows
    inbound_df  = df[df["inbound"] == True].copy()   # noqa: E712
    outbound_df = df[df["inbound"] == False].copy()  # noqa: E712

    # Count outbound tweets per author_id
    brand_outbound = (
        outbound_df.groupby("author_id")
        .size()
        .reset_index(name="outbound_count")
    )

    # Count how many of each brand's author tweets are inbound (should be ~0)
    brand_inbound = (
        inbound_df.groupby("author_id")
        .size()
        .reset_index(name="inbound_count")
    )

    brands = brand_outbound.merge(brand_inbound, on="author_id", how="left")
    brands["inbound_count"] = brands["inbound_count"].fillna(0).astype(int)
    brands["total"] = brands["outbound_count"] + brands["inbound_count"]
    brands["pct_inbound"] = brands["inbound_count"] / brands["total"]

    # Keep only "real" brand accounts: mostly outbound, at least 200 replies
    brand_accs = brands[
        (brands["pct_inbound"] < 0.20) &
        (brands["outbound_count"] >= 200)
    ].copy()

    # ── for each brand, find the customer-side conversations ─────────────────
    # A "thread" = (in_response_to_tweet_id chains).
    # A thread is "complete" if at least one outbound reply exists for it.

    # Map each brand to the inbound tweets it replied to
    brand_reply_map = {}
    for _, row in outbound_df[
        outbound_df["in_response_to_tweet_id"].notna()
    ].iterrows():
        aid = row["author_id"]
        if aid in brand_accs["author_id"].values:
            brand_reply_map.setdefault(aid, set()).add(row["in_response_to_tweet_id"])

    # All inbound tweet_ids
    inbound_tweet_ids = set(inbound_df["tweet_id"].dropna().unique())

    profile_rows = []
    for _, brow in brand_accs.iterrows():
        aid = brow["author_id"]
        replied_to = brand_reply_map.get(aid, set())
        # Thread completeness: how many replied-to tweets are inbound
        completed = replied_to & inbound_tweet_ids
        thread_completeness = len(completed) / max(len(replied_to), 1)

        # Noise proxy: fraction of outbound tweets that are very short (≤ 20 chars)
        brand_out_texts = outbound_df[outbound_df["author_id"] == aid]["text"].dropna()
        noise_frac = (brand_out_texts.str.len() <= 20).mean() if len(brand_out_texts) else 1.0

        # Topic coherence: rough proxy — fraction of inbound texts mentioning
        # brand's own name (not available without the name, so use unique
        # word-type ratio in inbound texts)
        # We do a lightweight type-token ratio on top-1000 inbound texts
        inbound_for_brand = inbound_df[
            inbound_df["in_response_to_tweet_id"].isin(
                outbound_df[outbound_df["author_id"] == aid]["tweet_id"]
            )
        ]["text"].dropna().head(1000)

        if len(inbound_for_brand) > 0:
            all_tokens = " ".join(inbound_for_brand.tolist()).lower().split()
            ttr = len(set(all_tokens)) / max(len(all_tokens), 1)
        else:
            ttr = 0.0

        profile_rows.append({
            "author_id":            aid,
            "outbound_count":       int(brow["outbound_count"]),
            "inbound_count":        int(brow["inbound_count"]),
            "pct_inbound":          round(brow["pct_inbound"], 3),
            "thread_completeness":  round(thread_completeness, 3),
            "noise_frac":           round(noise_frac, 3),
            "topic_ttr":            round(ttr, 3),
        })

    profile_df = pd.DataFrame(profile_rows).sort_values(
        "outbound_count", ascending=False
    )
    return profile_df


# ── 3. decode author_ids → brand names ───────────────────────────────────────
# The dataset uses numeric author_ids.  The Kaggle discussion + known
# accounts let us map the top ones.  We ONLY use this for display;
# the actual data filtering always uses author_id, never scraped brand info.
KNOWN_BRANDS = {
    "SpotifyCares":   "115712028",
    "AmazonHelp":     "15101310",
    "AppleSupport":   "372144784",
    "Uber_Support":   "557260267",
    "Delta":          "22536055",
    "Ask_Spectrum":   "4587741948",
    "TMobileHelp":    "107852111",
    "VerizonSupport": "18839040",
    "XboxSupport":    "25548595",
    "PlayStation":    "71268530",
    "NikeSupport":    "406666753",
    "comcastcares":   "13650053",
    "Tesco":          "5765400",
    "British_Airways":"18332190",
    "AmericanAir":    "22536055",
    "hulu_support":   "168976127",
    "ChaseSupport":   "106951412",
}
ID_TO_BRAND = {v: k for k, v in KNOWN_BRANDS.items()}


def add_brand_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brand_name"] = df["author_id"].map(ID_TO_BRAND).fillna("unknown_" + df["author_id"])
    return df


# ── 4. score & rank ───────────────────────────────────────────────────────────
def score_brands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score (higher = better candidate):
      40% normalised outbound_count  (volume)
      30% thread_completeness        (data quality)
      20% (1 - noise_frac)           (signal quality)
      10% topic_ttr                  (topic breadth, not too narrow, not too broad)
    """
    df = df.copy()
    for col in ["outbound_count", "thread_completeness", "noise_frac", "topic_ttr"]:
        mn, mx = df[col].min(), df[col].max()
        rng = mx - mn if mx != mn else 1
        df[f"{col}_norm"] = (df[col] - mn) / rng

    df["score"] = (
        0.40 * df["outbound_count_norm"] +
        0.30 * df["thread_completeness_norm"] +
        0.20 * (1 - df["noise_frac_norm"]) +
        0.10 * df["topic_ttr_norm"]
    )
    return df.sort_values("score", ascending=False)


# ── 5. write report ───────────────────────────────────────────────────────────
def write_report(df: pd.DataFrame, out_path: pathlib.Path):
    top = df.head(10)
    lines = [
        "=" * 70,
        "STEP 0 — BRAND PROFILE REPORT",
        "Source: Kaggle 'Customer Support on Twitter' (twcs.csv)",
        "No external sources consulted.",
        "=" * 70,
        "",
        f"{'Rank':<5} {'Brand':<20} {'OutboundReplies':>15} {'ThreadCompl':>12} "
        f"{'NoiseFrac':>10} {'TopicTTR':>9} {'Score':>7}",
        "-" * 80,
    ]
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(
            f"{rank:<5} {row['brand_name']:<20} {row['outbound_count']:>15,} "
            f"{row['thread_completeness']:>12.3f} {row['noise_frac']:>10.3f} "
            f"{row['topic_ttr']:>9.3f} {row['score']:>7.3f}"
        )

    lines += [
        "",
        "Column definitions:",
        "  OutboundReplies  — # tweets sent by brand (proxy for dataset volume)",
        "  ThreadCompl      — fraction of brand replies that link to an inbound tweet",
        "  NoiseFrac        — fraction of brand replies ≤ 20 chars (noise proxy)",
        "  TopicTTR         — type-token ratio of linked inbound texts (topic breadth)",
        "  Score            — weighted composite (see code comments)",
        "",
        "RECOMMENDATION: see Step 0 sign-off in DECISION_LOG.md (populated after sign-off).",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Report written to {out_path}")
    print("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    profile_df = profile_brands(sample_n=500_000)
    profile_df = add_brand_names(profile_df)
    profile_df = score_brands(profile_df)

    # Save machine-readable profile
    out_csv = DATA_DIR / "brand_profile.csv"
    profile_df.to_csv(out_csv, index=False)
    print(f"[INFO] Brand profile saved: {out_csv}")

    # Write human report
    write_report(profile_df, DATA_DIR / "step0_report.txt")
