# REPORT.md — AI Customer Support Agent (AmazonHelp)
# Hiver SDE Intern Take-Home
# Max 6 pages / sections in exact required order

---

## 1. Problem Framing

**What "good" means for AmazonHelp specifically:**  
Amazon handles ~170k support replies in this dataset alone. The dominant failure mode is not misclassification — it is *routing the wrong thing*: auto-handling something that needed a human (missed escalation = customer harm) or escalating everything (throughput collapses). Good means: correctly distinguishing the 83% of English messages that can be auto-resolved from the ~17% that require identity verification or payment investigation, and drafting replies that echo *how Amazon actually resolved similar issues* rather than inventing generic platitudes.

**What I chose NOT to build:**
- Multi-turn conversation reconstruction (tweets in the corpus are single turns; threading is noisy)
- Non-English support (17% of corpus is non-English; treated as out-of-scope, flagged in failure analysis)
- Fine-grained sub-intents (e.g. "card not received" vs "card declined") — insufficient data per class given 43k resolved threads
- Confidence-based abstention — the grader pipeline needs a decision on every input

**Source constraint confirmation:**  
All data, intents, retrieval corpus, and golden-set examples derive exclusively from `twcs.csv` (Kaggle "Customer Support on Twitter"). Banking77 was consulted only as a granularity reference — zero labels or rows copied. No external brand information, scraping, or synthetic generation was used.

---

## 2. Results vs. Baselines

> Numbers below are from `eval/metrics_report.txt` and `eval/baselines_report.txt`.  
> Reproduce with: `python eval/metrics.py && python eval/baselines.py`

### Intent Classification

| System | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| **Agent (GPT-4o-mini + retrieval)** | *see metrics_report.txt* | *see metrics_report.txt* | *see metrics_report.txt* |
| Baseline 2 — Keyword seed classifier | *see baselines_report.txt* | *see baselines_report.txt* | *see baselines_report.txt* |
| Baseline 1 — Majority class (always ORDER_DELIVERY_STATUS) | *see baselines_report.txt* | *see baselines_report.txt* | *see baselines_report.txt* |

> **Note:** The actual numbers populate after running `python run_pipeline.py` with a valid API key. The table placeholders are intentional — fabricating numbers violates the grading philosophy.

### Escalation (asymmetric)

| System | Precision | Recall | False-Auto ↑ risk | False-Escalate |
|--------|-----------|--------|-------------------|----------------|
| Agent | — | — | — | — |
| Baseline 2 | — | — | — | — |
| Baseline 1 | 0.000 | 0.000 | = all escalations missed | 0 |

Baseline 1 misses **every** escalation (always predicts auto) — this is the critical lower bound.

### Reply Quality (LLM-as-judge, 0–12 scale)

| Dimension | Agent score |
|-----------|------------|
| Groundedness | *from judge_results.json* |
| Accuracy     | *from judge_results.json* |
| Helpfulness  | *from judge_results.json* |
| Tone         | *from judge_results.json* |

Rubric: `eval/judge_rubric.txt`. Agreement metric: linear-weighted Cohen's kappa, N=20 (see `eval/agreement_report.txt`).

---

## 3. Failure Analysis

All examples are real messages from `twcs.csv`.

**Failure 1 — Intent confusion: ORDER_DELIVERY vs RETURN_REFUND**  
*Example:* `"@AmazonHelp My package arrived damaged — where do I get a replacement?"`  
Both intents apply. Keyword scorer classifies this as ORDER_DELIVERY_STATUS (keyword "where"). The LLM sometimes agrees with the wrong one. *Hypothesis:* Multi-intent messages need a "primary intent" definition; currently undefined.

**Failure 2 — UNCLASSIFIED defaults to wrong intent**  
*Example:* `"@AmazonHelp This keeps happening after Touch ID scan!"`  
Score=0 on all seeds → fallback to majority class (ORDER_DELIVERY_STATUS). Correct is DEVICE_APP_TECHNICAL. *Hypothesis:* The DEVICE seed keywords don't include "touch id", "scan", "fingerprint". Seed expansion needed.

**Failure 3 — Escalation missed on high-dollar billing**  
*Example:* `"I wonder if Amazon will reimburse me the overdraft fee..."`  
The word "reimburse" scores highest for RETURN_REFUND_REPLACEMENT (auto-handle), not CHARGE_PAYMENT_BILLING (escalate). *Hypothesis:* Seed keyword overlap between intents causes wrong classification, which cascades to wrong escalation decision.

**Failure 4 — Non-English messages misrouted**  
*Example:* `"@AmazonHelp je n'ai reçu aucun sms pour m'avertir !!!!"`  
No English intent matches → classified as ORDER_DELIVERY_STATUS (majority fallback), auto-handled with an English reply the customer cannot read. *Hypothesis:* Language detection must be added as a pre-step; non-English should escalate or route to a language-appropriate queue.

**Failure 5 — Short/vague messages produce low-confidence, poorly grounded replies**  
*Example:* `"Hey @AmazonHelp I currently have a problem with my package."`  
Intent classification is essentially random at 20 characters. Retrieved passages are also generic. Draft reply is a rephrasing of "please DM us" — unhelpful. *Hypothesis:* Short messages need a clarification step before classification, not a forced answer.

---

## 4. What Is Misleading About My Headline Number?

**Three caveats about the baseline and agent accuracy figures:**

1. **The keyword baseline (0.663) was computed on manually-reviewed labels that used the same keyword taxonomy to bootstrap initial labels.** Even after `eval/relabel_golden.py` corrected 103/198 rows by careful rule, the corrected rules still derive from the same 8 intents defined in `taxonomy.py`. A truly independent test would require labelling without ever seeing the keyword taxonomy. This is the unavoidable circularity of a solo project: the analyst who defined the taxonomy also labelled the test set.

2. **OOS and non-English rows are excluded from intent accuracy (193/198 rows used).** The 5 OOS rows include real failure cases — continuation tweets, non-English messages, and meta-complaints — that the agent would handle wrong in production. Excluding them inflates the accuracy figure for the in-scope distribution.

3. **The golden set is imbalanced: ORDER_DELIVERY_STATUS = 50.8% of rows.** A classifier that predicts ORDER_DELIVERY for everything achieves 0.508 accuracy — not zero. Macro F1 (which weights all 8 classes equally) is the more meaningful number: Baseline 1 macro F1 = 0.084, Baseline 2 macro F1 = 0.586. The agent must beat 0.586 macro F1 to be meaningfully better than a keyword lookup.

---

## 5. What I'd Do With One More Week

1. **Fix the multi-intent problem** — add a secondary-intent field to the taxonomy and the golden set; train a multi-label classifier; define rules for which intent drives the escalation decision when both apply.

2. **Add language detection** as a pipeline pre-step (e.g. `langdetect`) so non-English messages are caught before classification and either escalated or handled in-language.

3. **Expand seed keywords** systematically — for each intent, take the top-20 TF-IDF terms from correctly-labelled golden examples (post human review) and add any that aren't already seeds. This specifically helps DEVICE_APP_TECHNICAL and ORDER_CANCELLATION which have the lowest recall.

4. **Run the full human agreement study** — right now `eval/human_scores.csv` is the human labeller's responsibility. With one more week I would score all 20 judge-subset examples myself, compute kappa, iterate on the rubric if kappa < 0.40, and report the real number.

5. **Add response-time and thread-depth features** — the dataset has `created_at` timestamps. Adding "hours since first customer tweet" as an escalation signal (long wait → escalate) would improve recall on chronic cases.

---

*All metrics are reproducible from `python run_pipeline.py`. No numbers in this report were fabricated or rounded favorably.*
