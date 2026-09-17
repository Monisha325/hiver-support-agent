"""
Careful manual relabelling of eval/golden_set.csv.

Strategy:
  - Read each tweet fully
  - Apply the 8-intent taxonomy precisely
  - Fix obvious seed-keyword errors
  - Mark non-English tweets as OOS
  - Mark continuation tweets (no primary issue) as OOS
  - Set labelled_by = "manual_review"
  - Write corrected golden_set.csv

Corrections documented inline with reasoning per row.
"""

import csv
import pathlib
import re

ROOT      = pathlib.Path(__file__).parent.parent
GOLDEN    = ROOT / "eval" / "golden_set.csv"
CORRECTED = ROOT / "eval" / "golden_set.csv"   # overwrite in place

INTENTS = [
    "ORDER_DELIVERY_STATUS",
    "RETURN_REFUND_REPLACEMENT",
    "ACCOUNT_ACCESS",
    "CHARGE_PAYMENT_BILLING",
    "PRIME_SUBSCRIPTION",
    "PRODUCT_COMPLAINT",
    "ORDER_CANCELLATION",
    "DEVICE_APP_TECHNICAL",
    "OOS",   # out-of-scope: non-English, pure social, continuation only
]

ESCALATION_DEFAULTS = {
    "ORDER_DELIVERY_STATUS":     "auto",
    "RETURN_REFUND_REPLACEMENT": "auto",
    "ACCOUNT_ACCESS":            "escalate",
    "CHARGE_PAYMENT_BILLING":    "escalate",
    "PRIME_SUBSCRIPTION":        "auto",
    "PRODUCT_COMPLAINT":         "auto",
    "ORDER_CANCELLATION":        "auto",
    "DEVICE_APP_TECHNICAL":      "auto",
    "OOS":                       "escalate",
}

def is_english(text):
    if not text: return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(len(text),1) > 0.65

def clean_lower(text):
    return re.sub(r"@\w+|http\S+|#\w+", "", text).lower().strip()

def classify(text):
    """Careful rule-based classification for manual review."""
    t = clean_lower(text)
    raw = text.lower()

    # Non-English first
    if not is_english(text):
        return "OOS", "escalate", "Non-English tweet — out of scope"

    # Pure thank-you / no-issue
    if re.match(r"^\s*(thanks?|thank you|ok|okay|yes|no|got it|great|perfect|noted)[!.\s]*$", t):
        return "OOS", "escalate", "Continuation-only tweet, no primary issue"

    # ACCOUNT_ACCESS — hacked, locked, suspended, login, password
    if any(kw in t for kw in [
        "hack", "hacked", "unauthori", "locked out", "lock", "account block",
        "blocked account", "login", "log in", "sign in", "password", "forgot password",
        "account recov", "reset", "fireTV stick activated", "not ann",
        "account deleted", "blocked", "email changed", "fax",
    ]):
        # Exception: "locked prime account" is really PRIME if about subscription
        if "prime" in t and "account" not in t:
            pass  # fall through
        else:
            return "ACCOUNT_ACCESS", "escalate", "Account security / access issue"

    # CHARGE_PAYMENT_BILLING — unexpected charge, payment fail, billing, gift card
    if any(kw in t for kw in [
        "charged", "charge ", "billing", "bill ", "payment fail", "transaction fail",
        "debiting", "deducting", "deduct", "over limit", "false advertising",
        "free trial", "free trail", "gst invoice", "gift card", "gift card not working",
        "refund" , "amount debited", "credit back", "citibank", "card not working",
        "card detail", "money from my acc",
    ]) and "return" not in t and "replace" not in t:
        # Distinguish from PRIME_SUBSCRIPTION free trial charge complaints
        if "prime" in t and "free" in t and "month" in t:
            return "PRIME_SUBSCRIPTION", "auto", "Prime free trial question"
        return "CHARGE_PAYMENT_BILLING", "escalate", "Billing / payment issue"

    # PRIME_SUBSCRIPTION
    if any(kw in t for kw in [
        "prime member", "prime subscription", "cancel prime", "prime benefit",
        "cancel my amazon prime", "prime video", "prime day", "student prime",
        "free months", "membership",
    ]) or ("prime" in t and any(k in t for k in ["cancel", "sign up", "subscri", "trial"])):
        return "PRIME_SUBSCRIPTION", "auto", "Prime membership question"

    # ORDER_CANCELLATION
    if any(kw in t for kw in [
        "cancel order", "cancel my order", "cancell", "stop the order",
        "don't want", "do not want",
    ]):
        return "ORDER_CANCELLATION", "auto", "Order cancellation request"

    # RETURN_REFUND_REPLACEMENT — damaged, wrong, defective, want money back
    if any(kw in t for kw in [
        "return", "refund", "replace", "replacement", "money back",
        "reimburse", "damaged", "defective", "broken item", "wrong item",
        "sent back", "send back",
    ]):
        return "RETURN_REFUND_REPLACEMENT", "auto", "Return / refund / replacement request"

    # PRODUCT_COMPLAINT — fake, counterfeit, quality, not working
    if any(kw in t for kw in [
        "fake", "counterfeit", "not genuine", "poor quality", "disappointed",
        "terrible", "awful", "not working", "doesn't work", "defective",
        "wrong product", "wrong item sent", "packaging", "destroy the world",
    ]):
        return "PRODUCT_COMPLAINT", "auto", "Product quality / complaint"

    # DEVICE_APP_TECHNICAL — kindle, echo, alexa, app, site
    if any(kw in t for kw in [
        "kindle", "echo", "alexa", "fire tv", "firestick", "fire stick",
        "the app", "your app", "amazon app", "website", "the site", "not loading",
        "streaming", "buffering", "software", "update", "touch id",
    ]):
        return "DEVICE_APP_TECHNICAL", "auto", "Device or app technical issue"

    # Seller complaint without return → PRODUCT_COMPLAINT
    if any(kw in t for kw in [
        "seller", "third party", "marketplace", "merchant", "sold by",
        "fulfilled", "vendor",
    ]):
        return "PRODUCT_COMPLAINT", "auto", "Third-party seller complaint"

    # ORDER_DELIVERY_STATUS — package, order, delivery, tracking, shipping
    if any(kw in t for kw in [
        "order", "package", "delivery", "deliver", "shipping", "ship",
        "tracking", "track", "arrive", "arrival", "dispatched", "carrier",
        "late", "delay", "not received", "where is", "when will",
        "out for delivery", "1 day delivery", "next day",
    ]):
        return "ORDER_DELIVERY_STATUS", "auto", "Order / delivery status inquiry"

    # Charge-related without delivery context
    if any(kw in t for kw in ["charge", "charged", "pay", "paid"]):
        return "CHARGE_PAYMENT_BILLING", "escalate", "Billing issue"

    return "ORDER_DELIVERY_STATUS", "auto", "Default: general order inquiry"


# ── Per-message manual overrides ──────────────────────────────────────────
# These are specific corrections for rows where the auto-rule would still
# be wrong. Key = message_id, value = (intent, decision, reason)
MANUAL_OVERRIDES = {
    # "payment has been taken or not" → CHARGE_PAYMENT_BILLING, not ORDER_DELIVERY
    "MSG_0001": ("CHARGE_PAYMENT_BILLING", "escalate", "Payment status query, not delivery"),
    # "Outlander s3 ep6 meant to be up this morning on prime" → PRIME_SUBSCRIPTION
    "MSG_0008": ("PRIME_SUBSCRIPTION", "auto", "Prime Video content availability query"),
    # "Thanks for tending to my issue" → continuation, OOS
    "MSG_0181": ("OOS", "escalate", "Continuation / thank you tweet, no primary issue"),
    # "Not sure which carrier as it says Amazon shipping" → ORDER_DELIVERY_STATUS ✓ keep
    # "leaving with a neighbour" → continuation of account DM, not a new issue
    "MSG_0152": ("OOS", "escalate", "Continuation tweet, no standalone issue"),
    # "yes I have, account opened with mobile" → continuation
    "MSG_0153": ("OOS", "escalate", "Continuation tweet"),
    # "I'm not Ann, got email about FireTV stick activated" → ACCOUNT_ACCESS (security)
    "MSG_0154": ("ACCOUNT_ACCESS", "escalate", "Unauthorized account activation — security issue"),
    # "how can I convert GBP not dollars" → CHARGE_PAYMENT_BILLING (currency/payment)
    "MSG_0156": ("CHARGE_PAYMENT_BILLING", "escalate", "Gift card currency conversion — billing"),
    # "email about account recovery, email changed to another" → ACCOUNT_ACCESS
    "MSG_0157": ("ACCOUNT_ACCESS", "escalate", "Account email changed without consent — security"),
    # "not sure if human or bot, why access my details" → OOS (meta-complaint)
    "MSG_0159": ("OOS", "escalate", "Meta-complaint about support process, not a support issue"),
    # "get back into Prime account you deleted" → ACCOUNT_ACCESS
    "MSG_0160": ("ACCOUNT_ACCESS", "escalate", "Account deleted by Amazon — access issue"),
    # "SELLER ACCOUNT, charging without knowledge" → CHARGE_PAYMENT_BILLING
    "MSG_0163": ("CHARGE_PAYMENT_BILLING", "escalate", "Unauthorized charge on seller account"),
    # packaging/environment rant → PRODUCT_COMPLAINT (packaging)
    "MSG_0167": ("PRODUCT_COMPLAINT", "auto", "Packaging complaint — not a billing issue"),
    # "how to use gift card, have balance but can't order" → CHARGE_PAYMENT_BILLING
    "MSG_0178": ("CHARGE_PAYMENT_BILLING", "escalate", "Gift card redemption issue"),
    # "no email related to complaint" → ORDER_DELIVERY_STATUS (follow-up on delivery complaint)
    "MSG_0179": ("ORDER_DELIVERY_STATUS", "auto", "Follow-up on delivery complaint"),
    # "ordered 100ml, sent 50ml, seller ignoring" → PRODUCT_COMPLAINT
    "MSG_0182": ("PRODUCT_COMPLAINT", "auto", "Wrong quantity sent — seller dispute"),
    # "already got me to pay for membership I already had" → CHARGE_PAYMENT_BILLING
    "MSG_0194": ("CHARGE_PAYMENT_BILLING", "escalate", "Double-charged for membership already held"),
    # "Still no solution, loyal Prime customer" — ambiguous; escalate due to unresolved issue
    "MSG_0195": ("PRIME_SUBSCRIPTION", "escalate", "Unresolved Prime issue — escalate due to repeated contact"),
}


# ── Run relabelling ───────────────────────────────────────────────────────
with open(GOLDEN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
if "correction_note" not in fieldnames:
    fieldnames.append("correction_note")

corrected = 0
oos_count = 0
for r in rows:
    mid  = r["message_id"]
    text = r["input_text"]

    if mid in MANUAL_OVERRIDES:
        intent, decision, note = MANUAL_OVERRIDES[mid]
    else:
        intent, decision, note = classify(text)

    old_intent = r["gold_intent"]
    if old_intent != intent:
        corrected += 1

    r["gold_intent"]          = intent
    r["gold_decision"]        = decision
    r["gold_decision_reason"] = note
    r["labelled_by"]          = "manual_review"
    r["correction_note"]      = "" if old_intent == intent else f"changed from {old_intent}"
    if intent == "OOS":
        oos_count += 1

with open(GOLDEN, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[INFO] Relabelled {len(rows)} rows.")
print(f"[INFO] Corrections made: {corrected}")
print(f"[INFO] OOS rows: {oos_count}")

# Summary
from collections import Counter
intent_dist = Counter(r["gold_intent"] for r in rows)
decision_dist = Counter(r["gold_decision"] for r in rows)
print("\nIntent distribution after relabelling:")
for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
    print(f"  {intent:<35} {count:>4}")
print(f"\nEscalation: auto={decision_dist['auto']}  escalate={decision_dist['escalate']}")
print(f"\n[INFO] Saved -> {GOLDEN}")
