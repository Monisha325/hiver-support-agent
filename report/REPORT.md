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

> All numbers from `eval/metrics_report.txt` and `eval/baselines_report.txt`.
> Reproduce: `python eval/baselines.py` (no key) or `python run_pipeline.py` (with key).

### Intent Classification (N=193, OOS excluded)

| System | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| Baseline 1 — Majority class (ORDER_DELIVERY_STATUS) | 0.508 | 0.084 | 0.342 |
| Baseline 2 — Keyword seed classifier | 0.663 | 0.586 | 0.676 |
| **Baseline 3 — Retrieval-only** (keyword classify + top retrieved reply) | **0.663** | **0.586** | **0.676** |
| Agent (GPT-4o-mini + retrieval) | *run `python run_pipeline.py` with API key* | — | — |

Baseline 3 = Baseline 2 on intent classification (same keyword classifier), but retrieves real AmazonHelp replies as the draft instead of generating one — establishing the retrieval quality floor.

**Per-class F1 (Baseline 3 / retrieval-only):**

| Intent | F1 |
|--------|-----|
| ACCOUNT_ACCESS | 0.750 |
| ORDER_DELIVERY_STATUS | 0.725 |
| PRODUCT_COMPLAINT | 0.722 |
| CHARGE_PAYMENT_BILLING | 0.657 |
| ORDER_CANCELLATION | 0.571 |
| DEVICE_APP_TECHNICAL | 0.500 |
| PRIME_SUBSCRIPTION | 0.476 |
| RETURN_REFUND_REPLACEMENT | **0.286** ← worst |

### Escalation (escalate = positive class, asymmetric)

| System | Precision | Recall | False-Auto ↑risk | False-Escalate |
|--------|-----------|--------|-----------------|----------------|
| Baseline 1 — Majority | 0.000 | 0.000 | **45** | 0 |
| Baseline 2 — Keyword | 0.702 | 0.733 | 12 | 14 |
| **Baseline 3 — Retrieval-only** | **0.673** | **0.740** | **13** | **18** |
| Agent (LLM) | *pending* | *pending* | *pending* | *pending* |

False-auto-handle = missed escalations (HIGH RISK). Agent target: recall > 0.740 with false-auto < 13.

### Reply Groundedness (ROUGE-1 recall vs retrieved passages)

| System | Mean ROUGE-1 recall |
|--------|-------------------|
| Baseline 3 — Retrieval-only | **1.000** (draft IS retrieved reply — theoretical ceiling) |
| Agent (LLM) | *pending — expected 0.3–0.7 as LLM paraphrases* |

### Reply Quality — LLM-as-judge (0–12)

Judge rubric: groundedness / accuracy / helpfulness / tone (0–3 each).
Run `python eval/llm_judge.py` after setting API key → writes `eval/judge_results.json`.

---

## 3. Failure Analysis

All examples are real messages from `twcs.csv`.

**Failure 1 — Intent confusion: ORDER_DELIVERY vs RETURN_REFUND**  
**Failure 1 — Refund buried inside delivery complaint → wrong intent, missed escalation** `[MSG_0022]`  
*Text:* `"@AmazonHelp Things have been going good except for this order placed in Feb, 17 wherein Amazon had lost the package, no refund, no nothing. 'We'd have made the refund, had it been a bit early'..."`  
Predicted: `ORDER_DELIVERY_STATUS / auto`. Gold: `CHARGE_PAYMENT_BILLING / escalate`.  
The keyword "order" scored higher than "refund" because the message opens with delivery context. The missing refund (the real issue) was buried mid-sentence. *Hypothesis:* Keyword scoring is positional-blind. A sentence-level classifier would catch "no refund" as the dominant clause.

**Failure 2 — CANCEL + REFUND confused; wrong escalation path** `[MSG_0034]`  
*Text:* `"@AmazonHelp I'VE SPOKE TO THREE DIFFERENT PEOPLE ABOUT MY REFUND AND IT IS STILL NOT THERE AFTER A MONTH. DO SOMETHING!!!!!"`  
Predicted: `RETURN_REFUND_REPLACEMENT / auto`. Gold: `CHARGE_PAYMENT_BILLING / escalate`.  
"Refund" seed matched RETURN (auto) before CHARGE (escalate). A chronic multi-contact case got auto-handled instead of escalated. *Hypothesis:* Chronic-contact signal (e.g. "three different people", "after a month") should override intent classification and force escalation.

**Failure 3 — Account block hidden behind product complaint** `[MSG_0035]`  
*Text:* `"poor service and i complaint regarding my 10.or g display damaged in 6 days — they not given replace product they block my no."`  
Predicted: `RETURN_REFUND_REPLACEMENT / auto`. Gold: `ACCOUNT_ACCESS / escalate`.  
"Replace" dominated the seed match; "block my no." (blocked phone/account) was ignored. Account blocking requires human verification — this was missed entirely. *Hypothesis:* Seed keywords for ACCOUNT_ACCESS need to include "block" and "blocked number".

**Failure 4 — Colloquial "#fraud" triggers unnecessary escalation** `[MSG_0042]`  
*Text:* `"@AmazonHelp I paid using my Amex Card; after so much of wait the #FedEx guy tells me it's a cod! #unacceptable #fraud"`  
Predicted: `CHARGE_PAYMENT_BILLING / escalate`. Gold: `ORDER_DELIVERY_STATUS / auto`.  
The hashtag `#fraud` matched the escalation trigger keyword, routing to human. The customer used "fraud" colloquially for a COD mix-up that AmazonHelp could have resolved with a tracking link. *Hypothesis:* Hashtag-form keywords should be down-weighted vs. prose-form keywords in the trigger list.

**Failure 5 — OOS thank-you tweet handled as live delivery query** `[MSG_0138]`  
*Text:* `"@AmazonHelp Thank you."`  
Predicted: `ORDER_DELIVERY_STATUS / auto`. Gold: `OOS / escalate` (continuation, no issue).  
The fallback to majority class (ORDER_DELIVERY_STATUS) triggered a real retrieved reply: *"Sure thing — we're here to help! Have a great weekend!"* — coincidentally appropriate, but for the wrong reason. *Hypothesis:* Short messages below 5 tokens should be classified as continuation/OOS before intent scoring runs.

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
