# Intent Taxonomy — AmazonHelp
# Source: Derived from twcs.csv (Kaggle Customer Support on Twitter) ONLY
# Banking77 consulted ONLY as shape reference: ~77 fine-grained vs our 8 coarser intents
# No Banking77 labels, data rows, or examples appear here.
#
# Derivation method:
#   1. Extracted 50,284 inbound tweets that AmazonHelp replied to (full CSV)
#   2. Ran TF-IDF on 50k-row subsample to surface top terms
#   3. Grouped top terms into seed keyword lists; scored each tweet
#   4. Inspected UNCLASSIFIED bucket (43.9%) → found: non-English tweets,
#      device/app issues, social/vague replies, follow-up continuations
#   5. Consolidated 10 seeds into 8 final intents; folded small tail intents
#      (DELIVERY_ADDRESS_CHANGE, SELLER_THIRD_PARTY) into parent categories
#   6. Added DEVICE_APP_TECHNICAL for the Kindle/Echo/app cluster hidden in
#      unclassified; kept ESCALATION_CONTACT_REQUEST for the follow-up cluster
#
# Intents intentionally NOT created:
#   - Non-English content: treated as noise / out-of-scope for this build
#   - Social mentions / praise: not support intents
#   - DELIVERY_ADDRESS_CHANGE: absorbed into ORDER_DELIVERY_STATUS
#   - SELLER_THIRD_PARTY: absorbed into PRODUCT_COMPLAINT (most such tweets
#     are ultimately product/quality disputes)

INTENTS = [
    {
        "id": "ORDER_DELIVERY_STATUS",
        "label": "Order & Delivery Status",
        "description": (
            "Customer asking where their order is, requesting tracking info, "
            "reporting a late or missing delivery, or noting a carrier delay. "
            "Includes delivery address correction requests."
        ),
        "seed_keywords": [
            "where", "order", "package", "tracking", "delivery", "delivered",
            "arrive", "arrived", "expected", "ship", "shipping", "shipment",
            "status", "track", "dispatched", "carrier", "usps", "ups", "fedex",
            "estimated", "arrival", "late", "delayed", "delay", "not received",
            "missing package", "address", "change address", "wrong address",
            "update address", "redirect",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle: standard tracking lookup + ETA update. "
            "Escalate if >7 days past expected delivery with no resolution."
        ),
        "data_frequency_pct": 31.8 + 0.4,   # ORDER_DELIVERY_STATUS + DELIVERY_ADDRESS_CHANGE
    },
    {
        "id": "RETURN_REFUND_REPLACEMENT",
        "label": "Return, Refund & Replacement",
        "description": (
            "Customer requesting a return, refund, or replacement for a "
            "damaged, defective, or wrong item. Includes reimbursement requests "
            "and cases where a return label was already sent."
        ),
        "seed_keywords": [
            "return", "refund", "replacement", "exchange", "money back",
            "reimburse", "reimbursement", "credit", "sent back", "sending back",
            "damaged", "broken", "defective", "wrong item", "incorrect",
            "not as described", "replace", "send another",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle if standard policy applies. "
            "Escalate if customer reports repeated failed replacement or "
            "refund amount disputed."
        ),
        "data_frequency_pct": 4.6,
    },
    {
        "id": "ACCOUNT_ACCESS",
        "label": "Account Access & Security",
        "description": (
            "Customer cannot log in, has forgotten their password, had their "
            "account locked or suspended, or suspects unauthorized access / hacking."
        ),
        "seed_keywords": [
            "account", "login", "sign in", "password", "locked", "access",
            "verify", "verification", "email", "2fa", "two factor",
            "suspended", "banned", "hacked", "unauthorized", "forgot password",
            "reset", "cannot log", "can't log",
        ],
        "escalation_default": "escalate",
        "escalation_note": (
            "Always escalate: account security issues require identity verification "
            "that a bot cannot safely perform."
        ),
        "data_frequency_pct": 4.8,
    },
    {
        "id": "CHARGE_PAYMENT_BILLING",
        "label": "Charge, Payment & Billing Issue",
        "description": (
            "Customer reports an unexpected or incorrect charge, double billing, "
            "overdraft caused by Amazon, payment failure, or gift card / "
            "Amazon Pay issue."
        ),
        "seed_keywords": [
            "charge", "charged", "billing", "bill", "payment", "paid",
            "double charged", "overcharged", "invoice", "receipt",
            "card", "credit card", "debit", "unauthorized charge",
            "prime charge", "fee", "cost", "gift card", "amazonpay",
        ],
        "escalation_default": "escalate",
        "escalation_note": (
            "Escalate if amount > $50 or customer explicitly disputes the charge. "
            "Auto-handle only for informational billing queries."
        ),
        "data_frequency_pct": 2.9,
    },
    {
        "id": "PRIME_SUBSCRIPTION",
        "label": "Prime Membership & Subscription",
        "description": (
            "Customer has a question about Prime membership: cancellation, "
            "renewal, free-trial terms, Prime Video, Prime benefits, student "
            "Prime, or unexpected Prime charge."
        ),
        "seed_keywords": [
            "prime", "prime membership", "prime video", "prime day",
            "prime subscription", "free trial", "membership", "annual",
            "student prime", "cancel prime", "prime benefits",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle: explain cancellation steps or link to membership page. "
            "Escalate if customer reports being charged after cancellation."
        ),
        "data_frequency_pct": 3.2,
    },
    {
        "id": "PRODUCT_COMPLAINT",
        "label": "Product Quality & Seller Complaint",
        "description": (
            "Customer complains about product quality, receives a fake / "
            "counterfeit item, experiences a non-functional product, or has "
            "an issue with a third-party marketplace seller (fraud, wrong item, "
            "fulfilment failure)."
        ),
        "seed_keywords": [
            "quality", "fake", "counterfeit", "not working", "doesn't work",
            "broken", "poor quality", "disappointed", "terrible", "awful",
            "complaint", "issue with", "problem with", "defective product",
            "used", "opened", "expired", "wrong", "not genuine",
            "seller", "third party", "marketplace", "vendor", "sold by",
            "fulfilled", "independent seller", "merchant", "fraudulent seller",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle with standard return/refund offer. "
            "Escalate if customer alleges fraud or health/safety risk."
        ),
        "data_frequency_pct": 1.9 + 0.6,   # PRODUCT_QUALITY + SELLER_THIRD_PARTY
    },
    {
        "id": "ORDER_CANCELLATION",
        "label": "Order Cancellation",
        "description": (
            "Customer explicitly requests cancellation of an existing order "
            "before shipment, or reports that a cancellation request was not "
            "acted on."
        ),
        "seed_keywords": [
            "cancel", "cancelled", "cancellation", "cancel order",
            "stop order", "don't want", "do not want", "withdraw",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle if order is pre-shipment. "
            "Escalate if already shipped — becomes a return request."
        ),
        "data_frequency_pct": 1.2,
    },
    {
        "id": "DEVICE_APP_TECHNICAL",
        "label": "Device & App Technical Issue",
        "description": (
            "Customer reports a technical problem with an Amazon device (Kindle, "
            "Echo, Fire TV) or the Amazon app / website: crashes, login errors "
            "on the app, streaming issues on Prime Video, etc."
        ),
        "seed_keywords": [
            "kindle", "echo", "alexa", "fire tv", "firestick", "fire stick",
            "app", "website", "site", "crash", "crashing", "not loading",
            "streaming", "buffering", "prime video", "technical", "tech",
            "software", "update", "bug",
        ],
        "escalation_default": "auto",
        "escalation_note": (
            "Auto-handle with standard troubleshooting steps. "
            "Escalate if device replacement is required."
        ),
        "data_frequency_pct": None,   # hidden in UNCLASSIFIED; estimated ~3-5%
    },
]

# ── Shape reference note (Banking77) ──────────────────────────────────────
# Banking77 has 77 fine-grained intents for a single-domain (banking) product.
# We chose 8 broader intents because:
#   1. Amazon support is multi-domain; fine-graining to 77 would require
#      significantly more golden-set data per class.
#   2. The dataset's reply corpus is ~43k threads; 8 intents gives ~5k
#      threads/class on average, sufficient for retrieval grounding.
#   3. Banking77 granularity (e.g. separate intents for "card not working"
#      vs "card stolen" vs "card arrival") was used as a CALIBRATION SIGNAL:
#      we deliberately kept our intents broader than that, not finer.
#   No Banking77 labels, intent names, or data rows appear above.

# ── Derived convenience dicts (importable) ────────────────────────────────
INTENT_NAMES        = [i["id"]                   for i in INTENTS]
INTENT_DESCRIPTIONS = {i["id"]: i["description"] for i in INTENTS}
ESCALATION_DEFAULTS = {i["id"]: i["escalation_default"] for i in INTENTS}
ESCALATION_NOTES    = {i["id"]: i["escalation_note"]    for i in INTENTS}

if __name__ == "__main__":
    print(f"Defined {len(INTENTS)} intents:")
    for i in INTENTS:
        freq = f"{i['data_frequency_pct']:.1f}%" if i['data_frequency_pct'] else "~3-5% (est.)"
        print(f"  [{i['id']:<28}]  escalation={i['escalation_default']:<8}  freq={freq}")
