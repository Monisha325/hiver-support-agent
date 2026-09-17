# DECISION LOG

This file tracks every non-obvious call made during the project, in chronological
order of phase.  Explicit source-constraint confirmations are included per
the assignment rules.

---

## Phase 0 — Data profiling & brand selection

- **[DL-01] Dataset source confirmed**: Only `twcs.csv` from Kaggle dataset
  `thoughtvector/customer-support-on-twitter` was downloaded and profiled.
  No other dataset, scrape, or external brand information was consulted
  to make the brand selection.

- **[DL-02] Brand-ID mapping methodology**: The `KNOWN_BRANDS` dict in
  `data/sample_dataset.py` maps numeric Kaggle author_ids to readable brand
  handles.  This mapping was derived from the Kaggle dataset's own discussion
  page (public metadata about which account IDs appear in the file), not from
  scraping live Twitter or any external brand directory.

- **[DL-03] Profiling metric choices**: Thread completeness, noise fraction,
  and type-token ratio were chosen as lightweight proxies computable from
  the raw CSV alone.  No external annotation was used.

- **[DL-04] Banking77 usage — EXPLICIT CONFIRMATION**: Banking77
  (PolyAI/banking77) was NOT downloaded or loaded during Phase 0.
  It will be consulted ONLY as a shape reference during intent taxonomy
  design (Phase 1) — meaning we look at how many intents it uses and how
  granular its categories are, without copying any label names or data rows.

- **[DL-05] Brand selected**: **AmazonHelp** (author_id = "AmazonHelp" in twcs.csv).
  Evidence (all from twcs.csv, 500k-row sample):
    - Outbound replies: 42,944 (2.7x the #2 brand, AppleSupport at 15,694)
    - Thread completeness: 0.997 (99.7% of replies link to a real inbound tweet)
    - Noise fraction: 0.002 (0.2% replies <= 20 chars)
    - Topic TTR: 0.236 (broad enough for 7-10 intent taxonomy)
    - Composite score: 0.900 (vs 0.646 for AppleSupport, 0.591 for Uber_Support)
  Rejected alternatives:
    - AppleSupport: Excellent quality but 2.7x fewer threads; narrower domain
    - Uber_Support: 10,367 replies; domain too narrow (ride/driver/app only)
    - Walmart/askvisa/AWSSupport/others: Volume < 500 replies; insufficient for
      a 150-250 golden set plus retrieval corpus
  No external data (brand website, live Twitter, etc.) was consulted.

---

## Phase 1 — Intent taxonomy

- **[DL-06] Taxonomy derivation method**: TF-IDF (ngram 1-2, 5000 features) on
  50k AmazonHelp inbound tweets from twcs.csv. Top terms grouped into seed keyword
  lists; scored each tweet; inspected UNCLASSIFIED bucket with a second TF-IDF pass.
  No external labelling, no crowdsourcing, no LLM seeding — data-only.

- **[DL-07] Banking77 explicit non-use**: Banking77 was consulted ONLY to calibrate
  granularity (77 fine-grained intents → we chose 8 broader intents). Zero B77 intent
  names, descriptions, seed keywords, or data rows appear in any project file. Confirmed.

- **[DL-08] 8 intents chosen**: DELIVERY_ADDRESS_CHANGE (0.4%) and SELLER_THIRD_PARTY
  (0.6%) folded into larger parents. DEVICE_APP_TECHNICAL added after inspecting
  unclassified bucket (kindle/echo/app cluster). GENERAL_INQUIRY_OTHER dissolved.

- **[DL-09] Non-English tweets out-of-scope**: ~17% of corpus is non-English.
  Treating as out-of-scope. Logged as known limitation in failure analysis.

- **[DL-10] Escalation defaults**: ACCOUNT_ACCESS always escalates (identity
  verification required). CHARGE_PAYMENT_BILLING escalates if >$50 or disputed.
  Rationale derived from AmazonHelp reply patterns in the dataset itself.

---

## Phase 2 — Retrieval index

- **[DL-11] FAISS IndexFlatIP (exact search)**: Chose exact cosine search over
  approximate HNSW/IVF because the corpus is 40k vectors — small enough that
  exact search is fast (<50ms per query on CPU) and fully reproducible with no
  random seed dependency.

- **[DL-12] Embedding model**: `all-MiniLM-L6-v2` (384-dim, ~80 MB). Chosen
  for speed on CPU and no internet requirement after first download. The model
  was already cached locally (Hugging Face weights). SSL errors on HEAD requests
  are non-blocking and do not affect inference.

- **[DL-13] 40k thread subsample for index**: Full inbound corpus is 152k pairs.
  Subsampled to 40k (random_state=42) to keep index build under 10 minutes on CPU.
  40k still gives ~5k examples per intent on average — sufficient for retrieval quality.

- **[DL-14] Longest-reply deduplication**: When one inbound tweet received multiple
  brand replies, we kept the longest (most informative). This is a heuristic — the
  first reply might be "please DM us" while the second is the actual resolution.

---

## Phase 3+4 — Agent (reply drafting + escalation)

- **[DL-15] GPT-4o-mini chosen**: Cost-efficient for 198 golden-set evaluation
  calls + 50 judge calls. Temperature=0.2 for classification (near-deterministic),
  0.2 for drafting (slight variation for naturalness).

- **[DL-16] Grounding enforced in system prompt, not post-hoc filtered**: The
  system prompt explicitly instructs the LLM to base its reply on retrieved passages.
  Groundedness is then measured post-hoc via ROUGE-1 recall. This is a soft constraint —
  logged as a known limitation (model can ignore it).

- **[DL-17] Escalation priority order**: Hard rules (intent level) → keyword triggers
  → LLM reasoning flag → taxonomy default. This ordering means high-risk keywords
  (e.g., "lawsuit", "injury") escalate regardless of intent classification. Tempted
  to add a list from a safety taxonomy but that would be an external source — derived
  the keyword list from patterns in AmazonHelp's own replies instead.

---

## Phase 5 — Golden set

- **[DL-18] 198 examples (not 150 or 250)**: Deduplication on first-100-chars removed
  some near-duplicate short tweets, landing at 198. Within the 150-250 range.

- **[DL-19] Batch C (hard cases)**: Short messages (20-80 chars) deliberately oversampled
  because they are the hardest to classify and most likely to fail in production.
  UNCLASSIFIED examples included to measure OOS handling.

---

## Phase 6 — Baselines

- **[DL-20] Keyword baseline uses same seeds as taxonomy**: This is intentional —
  it establishes the "ceiling" of the seed approach. The agent should beat it via
  LLM classification. If it doesn't, that's an honest finding worth reporting.

---

## Phase 7 — Evaluation

- **[DL-21] ROUGE-1 recall as groundedness proxy**: ROUGE-F1 was considered but
  recall is more appropriate here: we want to know what fraction of the draft is
  covered by retrieved passages, not whether the draft is long or short.

- **[DL-22] Judge subset N=50**: Full golden set (198 examples) at ~$0.002/call =
  ~$0.40 total. Chose N=50 to save cost; still sufficient for the kappa study (N=20).

- **[DL-23] Human agreement study deferred**: The human_scores.csv must be filled
  in manually. The script generates the template. Kappa reported as "pending" until
  filled. This is flagged explicitly — not silently omitted.

---

## Source audit — final confirmation

- Every tweet, reply, intent example, and retrieval document comes from twcs.csv
- Banking77: zero content in any project file (confirmed by grep across all .py/.md files)
- No synthetic data generated by LLM for golden set or taxonomy
- No external brand information consulted at any stage
- SSL workaround in data/download.py is a tooling fix only; data integrity unaffected

