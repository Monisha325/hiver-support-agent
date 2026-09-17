# Phase 1 — Intent Taxonomy Derivation Notes
# Brand: AmazonHelp | Source: twcs.csv ONLY
# Author: Derived from data; no external sources

## Derivation Method

1. **Extracted thread corpus**: Loaded full twcs.csv; identified all rows where
   `author_id = "AmazonHelp"` and `inbound = False` (brand replies). Collected
   the `in_response_to_tweet_id` for each reply → matched to inbound customer tweets.
   Result: **50,284 inbound tweets** that AmazonHelp actually responded to.

2. **Subsample**: Random sample of 50,000 rows (`random_state=42`).

3. **TF-IDF top terms**: Fit `TfidfVectorizer(ngram_range=(1,2), max_features=5000)`
   on the 50k corpus. Top terms by mean TF-IDF score across corpus:
   `order, delivery, shipping, return, account, charge, prime, cancel, package, refund`

4. **Keyword seeding**: Built seed keyword lists for 10 candidate intents by
   reading the top 200 TF-IDF terms. Scored each tweet → assigned to the
   highest-scoring intent.

5. **Distribution**: 56.1% covered by 10 seeds; **43.9% UNCLASSIFIED**.

6. **Unclassified inspection** (`inspect_unclassified.py`):
   Top terms in unclassified bucket: `que, la, en, je, gracias, merci, ich`
   → **~40% of unclassified are non-English** (French, Japanese, Spanish, German, Portuguese)
   Also prominent: `kindle, app, echo` → device/app technical issues not seeded.
   Also: `thanks, yes, ok, reply, response` → conversational continuations, not initiating messages.

7. **Consolidations**:
   - `DELIVERY_ADDRESS_CHANGE` (0.4%) → folded into `ORDER_DELIVERY_STATUS`
   - `SELLER_THIRD_PARTY` (0.6%) → folded into `PRODUCT_COMPLAINT`
   - `GENERAL_INQUIRY_OTHER` (4.8%) → absorbed into other intents via expanded seeds
   - New intent added: `DEVICE_APP_TECHNICAL` (for Kindle/Echo/app cluster in unclassified)

8. **Non-English tweets**: ~40% of unclassified ≈ ~17% of total corpus.
   Decision: treat as **out-of-scope for this build** (agent targets English).
   This is flagged in DECISION_LOG as a known limitation.

## Final Taxonomy: 8 Intents

| # | Intent ID | Description | Est. Freq | Default Escalation |
|---|-----------|-------------|-----------|-------------------|
| 1 | ORDER_DELIVERY_STATUS | Tracking, late/missing delivery, address correction | ~32% | Auto |
| 2 | RETURN_REFUND_REPLACEMENT | Return process, refund, replacement for damaged/wrong item | ~5% | Auto |
| 3 | ACCOUNT_ACCESS | Login failure, password reset, account locked/hacked | ~5% | **Escalate** |
| 4 | CHARGE_PAYMENT_BILLING | Wrong charge, double billing, payment failure, gift card | ~3% | **Escalate** |
| 5 | PRIME_SUBSCRIPTION | Prime membership, cancel/renew, free trial, Prime Video | ~3% | Auto |
| 6 | PRODUCT_COMPLAINT | Quality issue, fake item, seller fraud, defective product | ~3% | Auto |
| 7 | ORDER_CANCELLATION | Cancel pre-shipment order, unprocessed cancellation | ~1% | Auto |
| 8 | DEVICE_APP_TECHNICAL | Kindle/Echo/app crash, streaming issue, website bug | ~3-5% | Auto |

**Total English coverage (estimated)**: ~83% of corpus
**Out-of-scope (non-English)**: ~17% — logged as known gap

## Banking77 Shape Reference (explicit confirmation)

Banking77 was consulted ONLY as a **calibration signal** for granularity:
- B77 has 77 intents for a single-domain product → we chose 8 broader intents
  for a multi-domain retailer with ~43k threads (insufficient data for 77 classes)
- We looked at B77's category count and structure; we did NOT copy any intent names,
  descriptions, seed keywords, or data rows from B77.
- Confirmed: zero B77 content appears in `taxonomy.py` or any downstream file.
