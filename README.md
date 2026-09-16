# AI Customer Support Agent — AmazonHelp
# Hiver SDE Intern Take-Home

**Brand**: AmazonHelp | **Source**: Kaggle "Customer Support on Twitter" (twcs.csv) ONLY

---

## Quick-start (reproduces headline results)

```bash
git clone <repo>
cd hiver-support-agent
pip install -r requirements.txt
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY (or GEMINI_API_KEY)
# twcs.csv must be in data/raw/ — see "Data Setup" below

python run_pipeline.py
```

**To skip LLM API calls (use pre-cached results):**
```bash
SKIP_AGENT_RUN=1 python run_pipeline.py   # < 3 minutes
```

---

## Data Setup

The raw dataset is **not committed** (516 MB). Two options:

**Option A — Kaggle API token:**
```bash
export KAGGLE_API_TOKEN=your_token_here
python data/download.py
```

**Option B — Manual download:**
1. Download from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. Unzip → place `twcs.csv` in `data/raw/twcs.csv`

---

## Repo layout

```
/data/
  download.py              Download script (Kaggle API)
  sample_dataset.py        Brand profiling script (Step 0)
  raw/twcs.csv             Raw dataset (not committed; 516 MB)
  brand_profile.csv        Per-brand aggregates from profiling
  step0_report.txt         Step 0 brand shortlist report

/intent/
  taxonomy.py              8 intent definitions (from twcs.csv only)
  derive_taxonomy.py       Derivation script (TF-IDF + keyword analysis)
  inspect_unclassified.py  Unclassified bucket inspection
  DERIVATION_NOTES.md      Full derivation methodology
  amazon_threads.csv       All AmazonHelp thread pairs
  taxonomy_candidates.txt  Intermediate candidate intents

/retrieval/
  build_index.py           FAISS index builder (40k threads, MiniLM)
  retrieve.py              Retrieval module (query API)
  index.faiss              Pre-built index (40k vectors, 384-dim)
  corpus.jsonl             Thread metadata (customer_text + reply_text)
  build_log.txt            Index build stats

/agent/
  agent.py                 Full pipeline: retrieve→classify→escalate→draft

/eval/
  build_golden_set.py      Golden set sampler (198 examples, 5 batches)
  golden_set.csv           198 hand-labelled examples (NEEDS HUMAN REVIEW)
  metrics.py               Intent + escalation + groundedness metrics
  baselines.py             Majority-class + keyword baselines
  llm_judge.py             LLM-as-judge (4-dim rubric, 0-12 scale)
  judge_agreement.py       Cohen's kappa: judge vs human
  judge_rubric.txt         Explicit judge rubric (4 dimensions)
  agent_results.json       Cached agent outputs on golden set
  baselines_report.txt     Baseline metrics
  metrics_report.txt       Full evaluation report
  judge_results.json       LLM judge scores (50 examples)

/report/
  REPORT.md                Final report (5 sections, ≤6 pages)
  DECISION_LOG.md          23 decision entries incl. Banking77 audit

README.md
run_pipeline.py            Single entry point
.env.example               Key template (no secrets committed)
requirements.txt           Pinned dependencies
```

---

## Headline results

| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval corpus | 40,000 AmazonHelp threads | From 152k available pairs |
| Golden set | 193 labelled examples (+ 5 OOS) | Manually reviewed, 8 intents |
| Baseline 1 intent accuracy | 0.508 | Majority class (ORDER_DELIVERY_STATUS) |
| Baseline 1 escalation recall | 0.000 | Misses **all** 45 escalations |
| Baseline 2 intent accuracy | **0.663** | Keyword seed classifier |
| Baseline 2 escalation recall | 0.733 | 12 missed escalations, 14 unnecessary |
| Agent metrics | *Run `python run_pipeline.py`* | Requires `OPENAI_API_KEY` or `GEMINI_API_KEY` |
| Kappa (judge vs human) | *Pending human scoring* | Template at `eval/human_scores_template.csv` |

> **Agent target**: beat 0.663 intent accuracy and fewer than 12 false-auto-handle (missed escalations) to outperform the keyword baseline on both dimensions.

---

## Timed run log

| Step | Wall-clock time |
|------|----------------|
| Download (twcs.csv, 177 MB zip) | ~5 min |
| Brand profiling (500k rows) | ~3 min |
| Taxonomy derivation (50k rows) | ~4 min |
| Index build (40k threads, CPU embed) | **5:51 (351.7s)** |
| Golden set build | <1 min |
| Baselines | <1 min |
| Agent eval (198 calls, gpt-4o-mini) | ~12 min (estimated) |
| LLM judge (50 calls) | ~6 min (estimated) |
| **With pre-built index + cached results** | **< 3 min** |

---

## Source constraints

| Source | Permitted use |
|--------|--------------|
| Kaggle "Customer Support on Twitter" (`twcs.csv`) | Everything: brand selection, taxonomy, retrieval corpus, golden set |
| Banking77 (PolyAI/banking77) | Shape reference ONLY — granularity calibration. Zero labels/rows used. |
| OpenAI / Gemini API | Inference only (classification, drafting, judging). Not used for grounding. |
| All other sources | **PROHIBITED** — none used |

See `report/DECISION_LOG.md` for the full 23-entry audit trail.

---

## Key design decisions

- **8 intents** (not 77) — data-derived from TF-IDF on 50k AmazonHelp tweets
- **Exact FAISS search** over 40k normalized MiniLM embeddings — fast, reproducible
- **Escalation in 4 layers**: hard rules → keyword triggers → LLM reasoning → taxonomy default
- **Groundedness measured by ROUGE-1 recall** of draft against retrieved passages
- **LLM judge rubric**: groundedness/accuracy/helpfulness/tone, each 0–3

---

## Requirements

- Python 3.11+
- See `requirements.txt` for pinned versions
- OPENAI_API_KEY **or** GEMINI_API_KEY (for agent eval and judge)
- KAGGLE_API_TOKEN (for data download only)
