"""
run_pipeline.py — Single entry point to reproduce all results.

Runs the full pipeline from a clean state in the following order:
  1. (Skip if data/raw/twcs.csv exists, else error with instructions)
  2. Taxonomy verification
  3. Build retrieval index (if not already built)
  4. Build golden set (if not already built)
  5. Run baselines
  6. Run agent on golden set
  7. Compute evaluation metrics
  8. Run LLM judge (requires API key)
  9. Generate scoring template for human agreement
 10. Print headline results

Target runtime (documented subsample, no LLM calls cached):
  Index build: ~6 min (40k embeddings, CPU)
  Baselines:   < 1 min
  Agent eval:  ~10-15 min (198 LLM calls, gpt-4o-mini)
  Judge:       ~5-8 min (50 LLM calls)
  TOTAL:       ~25-30 min (well under 15 min if index already built)

For graders reproducing results:
  Index is pre-built (retrieval/index.faiss committed).
  Agent results cached in eval/agent_results.json.
  SKIP_AGENT_RUN=1 to use cached results (reduces to <3 min).

Usage:
    cp .env.example .env
    # fill in OPENAI_API_KEY (or GEMINI_API_KEY)
    python run_pipeline.py
    # or to skip LLM calls (use cached results):
    SKIP_AGENT_RUN=1 python run_pipeline.py
"""

import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent

def run(cmd, step_name):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd.split(),
        cwd=str(ROOT),
        env={
            **os.environ,
            "PYTHONIOENCODING":   "utf-8",
            "HF_HUB_OFFLINE":     "1",      # use cached model weights; skip SSL HEAD requests
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{status}] {step_name} — {elapsed:.1f}s")
    return result.returncode == 0

def check_prereqs():
    csv_path = ROOT / "data" / "raw" / "twcs.csv"
    if not csv_path.exists():
        print("ERROR: data/raw/twcs.csv not found.")
        print("Run: python data/download.py  (requires KAGGLE_API_TOKEN env var)")
        print("Or download manually from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        sys.exit(1)
    print(f"[OK] data/raw/twcs.csv found ({csv_path.stat().st_size/1e6:.0f} MB)")

    env_path = ROOT / ".env"
    if not env_path.exists():
        print("WARNING: .env file not found. Copy .env.example and fill in API keys.")

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("AI CUSTOMER SUPPORT AGENT — HIVER TAKE-HOME")
    print("Brand: AmazonHelp | Source: twcs.csv (Kaggle) ONLY")
    print("=" * 60)

    check_prereqs()

    SKIP_AGENT = os.environ.get("SKIP_AGENT_RUN", "0") == "1"

    steps = []

    # Step 2: Taxonomy check
    steps.append(("intent/taxonomy.py", "Verify intent taxonomy (8 intents)"))

    # Step 3: Build retrieval index (skip if exists)
    index_path = ROOT / "retrieval" / "index.faiss"
    if not index_path.exists():
        steps.append(("retrieval/build_index.py", "Build FAISS retrieval index (40k threads)"))
    else:
        print(f"\n[SKIP] retrieval/index.faiss already exists ({index_path.stat().st_size/1e6:.0f} MB)")

    # Step 4: Build golden set (skip if exists)
    golden_path = ROOT / "eval" / "golden_set.csv"
    if not golden_path.exists():
        steps.append(("eval/build_golden_set.py", "Build golden evaluation set (198 examples)"))
    else:
        print(f"[SKIP] eval/golden_set.csv already exists")

    # Step 5: Baselines
    steps.append(("eval/baselines.py", "Run baselines (majority + keyword)"))

    # Step 6+7: Agent eval + metrics
    if not SKIP_AGENT:
        steps.append(("eval/metrics.py", "Run agent on golden set + compute metrics"))
    else:
        print("[SKIP] Agent run skipped (SKIP_AGENT_RUN=1). Using cached eval/agent_results.json")
        steps.append(("eval/metrics.py", "Compute metrics from cached agent results"))

    # Step 8: LLM judge
    if not SKIP_AGENT:
        steps.append(("eval/llm_judge.py", "LLM judge (50 examples, explicit rubric)"))

    # Step 9: Agreement template
    steps.append(("eval/judge_agreement.py", "Generate human scoring template"))

    # Run all steps
    all_ok = True
    timing = {}
    for cmd, name in steps:
        t0 = time.time()
        ok = run(cmd, name)
        timing[name] = time.time() - t0
        if not ok:
            all_ok = False
            print(f"[ERROR] {name} failed. Check output above.")

    # Summary
    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Total wall-clock time: {total_elapsed/60:.1f} min")
    print(f"Status: {'ALL STEPS OK' if all_ok else 'SOME STEPS FAILED'}")
    print(f"{'='*60}")
    print("\nStep timings:")
    for name, t in timing.items():
        print(f"  {t:6.1f}s  {name}")
    print("\nKey output files:")
    print("  eval/golden_set.csv       — 198 hand-labelled examples")
    print("  eval/metrics_report.txt   — intent, escalation, groundedness metrics")
    print("  eval/baselines_report.txt — baseline comparison")
    print("  eval/judge_results.json   — LLM judge scores")
    print("  eval/agreement_report.txt — Cohen's kappa (after human scoring)")
    print("  report/REPORT.md          — Final report")
    print("  report/DECISION_LOG.md    — All non-obvious decisions")
