"""
Phase 3 — Reply drafting agent.
Phase 4 — Escalation policy.

Given a customer message, this module:
  1. Classifies the intent (LLM with retrieval-augmented few-shot examples)
  2. Decides auto-handle vs escalate WITH a stated reason
  3. Drafts a reply grounded in AmazonHelp's real historical resolutions
     (retrieved from retrieval/index.faiss — twcs.csv only)

GROUNDING CONTRACT:
  - The system prompt always includes verbatim retrieved reply snippets
    from twcs.csv. The LLM is instructed to base its reply on those.
  - The drafted reply is NOT generated purely from pretrained knowledge.
  - The groundedness metric in eval/ measures how much of the draft
    overlaps with the retrieved passages (ROUGE-1 recall).

Usage:
    from agent.agent import run_agent
    result = run_agent("My package hasn't arrived for 10 days")
    print(result)  # dict: intent, decision, reason, draft_reply, retrieved_hits
"""

import os
import json
import pathlib
import sys

from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.retrieve import Retriever
from intent.taxonomy import INTENTS   # noqa: E402

# ── Load API key ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional; fall back to env vars

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")

# Prefer whichever key is set
if OPENAI_API_KEY:
    LLM_BACKEND = "openai"
elif GEMINI_API_KEY:
    LLM_BACKEND = "gemini"
elif GROQ_API_KEY:
    LLM_BACKEND = "groq"
else:
    LLM_BACKEND = None  # will raise at call time

INTENT_NAMES = [i["id"] for i in INTENTS]
INTENT_DESCRIPTIONS = {i["id"]: i["description"] for i in INTENTS}
ESCALATION_DEFAULTS = {i["id"]: i["escalation_default"] for i in INTENTS}
ESCALATION_NOTES    = {i["id"]: i["escalation_note"] for i in INTENTS}

_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ── Escalation policy (Phase 4) ───────────────────────────────────────────
ESCALATION_HARD_RULES = {
    # intent → always escalate regardless of content
    "ACCOUNT_ACCESS": True,
}

ESCALATION_TRIGGER_KEYWORDS = [
    # Phrases that force escalation even for auto-handle intents
    "lawyer", "lawsuit", "legal action", "police", "fraud", "scam",
    "identity theft", "stolen", "threatening", "unsafe", "injury",
    "hurt", "hospital", "dead", "dying", "health risk",
    "media", "press", "news", "bbc", "cnn",   # threat of public shaming
    "block", "blocked", "locked", "hacked",   # account access issues
]


def decide_escalation(intent: str, customer_text: str, llm_reasoning: str) -> tuple[str, str]:
    """
    Returns (decision, reason) where decision is 'auto' or 'escalate'.

    Rules (in priority order):
      1. Hard-rule intents → always escalate
      2. Trigger keywords in customer text → escalate
      3. LLM says escalate in its reasoning → escalate
      4. Default from taxonomy
    """
    text_lower = customer_text.lower()

    # Rule 1: hard-rule intents
    if ESCALATION_HARD_RULES.get(intent):
        return "escalate", (
            f"Intent '{intent}' always requires human agent — "
            f"identity verification cannot be safely automated. "
            f"({ESCALATION_NOTES[intent]})"
        )

    # Rule 2: trigger keywords (use word boundaries to avoid matching colloquial hashtags like #fraud)
    import re
    matched_kws = [kw for kw in ESCALATION_TRIGGER_KEYWORDS if re.search(rf'\b{kw}\b', text_lower)]
    if matched_kws:
        return "escalate", (
            f"Customer message contains high-risk keyword(s): {matched_kws}. "
            "Routing to human agent."
        )

    # Rule 3: LLM flagged escalation in its reasoning
    reasoning_lower = llm_reasoning.lower()
    if any(w in reasoning_lower for w in ["escalate", "human agent", "cannot resolve"]):
        return "escalate", (
            "LLM classification reasoning indicated escalation is warranted. "
            f"LLM note: {llm_reasoning[:200]}"
        )

    # Rule 4: taxonomy default
    default = ESCALATION_DEFAULTS.get(intent, "auto")
    reason  = ESCALATION_NOTES.get(intent, f"Default policy for {intent}.")
    return default, reason


# ── LLM call (OpenAI) ─────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_openai(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_gemini(prompt: str) -> str:
    import time
    time.sleep(4)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-3.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_groq(messages: list[dict], model: str = "qwen/qwen3.8-27b") -> str:
    import openai
    client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def _call_llm(system: str, user: str) -> str:
    if LLM_BACKEND == "openai":
        return _call_openai([
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])
    elif LLM_BACKEND == "gemini":
        return _call_gemini(f"{system}\n\nUser: {user}")
    elif LLM_BACKEND == "groq":
        return _call_groq([
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])
    else:
        raise RuntimeError(
            "No LLM API key set. Add OPENAI_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY to .env"
        )


# ── Classification prompt ──────────────────────────────────────────────────
def _build_classify_prompt(customer_text: str, hits: list[dict]) -> tuple[str, str]:
    intent_list = "\n".join(
        f"  {i['id']}: {i['description']}" for i in INTENTS
    )

    # Include retrieved similar cases as context
    retrieved_context = ""
    for j, h in enumerate(hits[:3], 1):
        retrieved_context += (
            f"\nSimilar past case {j} (similarity={h['score']:.2f}):\n"
            f"  Customer: {h['customer_text'][:200]}\n"
            f"  Amazon replied: {h['reply_text'][:200]}\n"
        )

    system = (
        "You are an intent classifier for Amazon customer support.\n"
        "Classify the customer message into EXACTLY ONE of these intents:\n"
        f"{intent_list}\n\n"
        "CRITICAL RULES:\n"
        "1. If a message contains multiple issues (e.g. delivery issue AND a refund request), prioritize the intent that involves payments, refunds, or account access (e.g., CHARGE_PAYMENT_BILLING).\n"
        "2. If a customer mentions their account or phone number is 'blocked' or 'locked', it is ALWAYS ACCOUNT_ACCESS.\n"
        "3. Ignore colloquial frustration hashtags like #fraud if the actual issue is just a delayed delivery.\n\n"
        "Output a JSON object with keys: "
        "'intent' (one of the IDs above), 'confidence' (0.0-1.0), "
        "'reasoning' (one sentence explaining why). "
        "Output ONLY valid JSON, no markdown."
    )
    user = (
        f"Customer message: {customer_text}\n"
        f"\nContext — similar resolved cases from Amazon's support history:{retrieved_context}"
    )
    return system, user


# ── Reply drafting prompt ──────────────────────────────────────────────────
def _build_reply_prompt(
    customer_text: str,
    intent: str,
    hits: list[dict],
    decision: str,
    reason: str,
) -> tuple[str, str]:

    retrieved_replies = "\n".join(
        f"  [{j}] {h['reply_text'][:300]}"
        for j, h in enumerate(hits[:5], 1)
    )

    system = (
        "You are an AmazonHelp customer support agent.\n"
        "Draft a reply to the customer's message.\n\n"
        "GROUNDING RULE: Your reply MUST be grounded in the retrieved historical "
        "Amazon replies shown below. Do not rely solely on general knowledge. "
        "Adapt the language and steps from those real replies to fit this situation.\n\n"
        f"Detected intent: {intent}\n"
        f"Escalation decision: {decision} — {reason}\n\n"
        "Guidelines:\n"
        "- Be concise (2-4 sentences), empathetic, and actionable\n"
        "- If escalating, tell the customer a human agent will follow up and why\n"
        "- Do NOT invent order numbers, tracking IDs, or specific dates\n"
        "- Do NOT promise things the retrieved resolutions don't support\n"
        "- Reference steps / information from the retrieved replies where applicable\n"
        "Output only the reply text, no metadata."
    )
    user = (
        f"Customer message: {customer_text}\n\n"
        f"Retrieved AmazonHelp historical replies (from real support conversations):\n"
        f"{retrieved_replies}"
    )
    return system, user


# ── Main agent entrypoint ──────────────────────────────────────────────────
def run_agent(customer_text: str, top_k: int = 5) -> dict:
    """
    Full pipeline: retrieve → classify → escalate → draft.

    Returns:
        {
          "customer_text":  str,
          "intent":         str,
          "confidence":     float,
          "decision":       "auto" | "escalate",
          "reason":         str,
          "draft_reply":    str,
          "retrieved_hits": list[dict],
          "llm_reasoning":  str,
        }
    """
    # Step 1: retrieve
    retriever = get_retriever()
    hits = retriever.query(customer_text, top_k=top_k)

    # Step 2: classify
    sys_p, usr_p = _build_classify_prompt(customer_text, hits)
    raw_classify  = _call_llm(sys_p, usr_p)
    try:
        import re
        match = re.search(r'\{.*\}', raw_classify, re.DOTALL)
        if match:
            raw_classify = match.group(0)
        classify_json = json.loads(raw_classify)
        intent     = classify_json.get("intent", "ORDER_DELIVERY_STATUS")
        confidence = float(classify_json.get("confidence", 0.5))
        reasoning  = classify_json.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback: extract intent from text
        intent, confidence, reasoning = "ORDER_DELIVERY_STATUS", 0.3, raw_classify[:200]

    # Validate intent against known list
    if intent not in INTENT_NAMES:
        intent = "ORDER_DELIVERY_STATUS"

    # Step 3: escalation decision (Phase 4)
    decision, reason = decide_escalation(intent, customer_text, reasoning)

    # Step 4: draft reply
    sys_p2, usr_p2 = _build_reply_prompt(customer_text, intent, hits, decision, reason)
    draft_reply     = _call_llm(sys_p2, usr_p2)

    return {
        "customer_text":  customer_text,
        "intent":         intent,
        "confidence":     confidence,
        "decision":       decision,
        "reason":         reason,
        "draft_reply":    draft_reply,
        "retrieved_hits": hits,
        "llm_reasoning":  reasoning,
    }


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "My order was supposed to arrive 3 days ago, still nothing",
        "I cannot log in, my password reset email never arrived",
        "I want to cancel my Prime membership, I'm being charged $139",
    ]
    for t in tests:
        print(f"\n{'='*60}")
        print(f"INPUT: {t}")
        res = run_agent(t)
        print(f"INTENT:    {res['intent']} (conf={res['confidence']:.2f})")
        print(f"DECISION:  {res['decision'].upper()}")
        print(f"REASON:    {res['reason']}")
        print(f"REPLY:\n{res['draft_reply']}")
