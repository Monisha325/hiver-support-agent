"""
Phase 7 — LLM-as-judge with explicit rubric.

For each agent result in eval/agent_results.json, asks an LLM judge
to score the draft reply on 4 dimensions with a fixed rubric.

Rubric (0–3 scale per dimension, total 0–12):
  1. GROUNDEDNESS (0-3): Is the reply grounded in the retrieved passages?
       3 = directly uses specific info from retrieved AmazonHelp replies
       2 = broadly consistent with retrieved context
       1 = partially grounded; some generic content
       0 = no apparent connection to retrieved passages
  2. ACCURACY (0-3): Is the information factually accurate given the context?
       3 = fully accurate; no hallucinated facts
       2 = mostly accurate; minor imprecision
       1 = some inaccurate or unsupported claims
       0 = significant hallucinations or wrong information
  3. HELPFULNESS (0-3): Does the reply address the customer's actual issue?
       3 = clearly resolves or escalates appropriately with next steps
       2 = addresses issue but missing one key element
       1 = tangentially relevant but not actionable
       0 = irrelevant or unhelpful
  4. TONE (0-3): Is the reply professional, empathetic, and concise?
       3 = professional, warm, appropriately brief
       2 = acceptable tone; minor issues
       1 = overly formal/robotic or slightly inappropriate
       0 = rude, dismissive, or excessively long

Judge outputs JSON: {groundedness, accuracy, helpfulness, tone, overall_comment}

Usage:
    python eval/llm_judge.py
    (reads eval/agent_results.json, writes eval/judge_results.json)
"""

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_JSON = ROOT / "eval" / "agent_results.json"
JUDGE_JSON   = ROOT / "eval" / "judge_results.json"
RUBRIC_FILE  = ROOT / "eval" / "judge_rubric.txt"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

RUBRIC_TEXT = """
You are an expert customer support quality evaluator.

Score the following AI-drafted reply on 4 dimensions (0-3 each):

GROUNDEDNESS (0-3):
  3 = directly uses specific info/steps from the retrieved AmazonHelp replies shown
  2 = broadly consistent with retrieved context
  1 = partially grounded; mix of retrieved and generic content
  0 = no apparent connection to the retrieved passages

ACCURACY (0-3):
  3 = no hallucinated facts; all claims supported by context or general policy
  2 = mostly accurate; minor imprecision
  1 = some unsupported claims
  0 = significant hallucinations or wrong information

HELPFULNESS (0-3):
  3 = clearly resolves the issue or escalates with clear next steps
  2 = addresses issue but missing one key actionable element
  1 = tangentially relevant but not actionable
  0 = irrelevant or unhelpful

TONE (0-3):
  3 = professional, empathetic, appropriately concise
  2 = acceptable; minor issues
  1 = overly robotic, slightly dismissive, or too long
  0 = rude, inappropriate, or extremely long/short

Output ONLY a JSON object with keys:
  groundedness (int 0-3), accuracy (int 0-3), helpfulness (int 0-3),
  tone (int 0-3), total (int 0-12), overall_comment (one sentence)
No markdown, no explanation outside the JSON.
""".strip()


def call_judge(customer_text, retrieved_replies, draft_reply, intent, decision):
    retrieved_str = "\n".join(
        f"  [{i+1}] {h['reply_text'][:300]}"
        for i, h in enumerate(retrieved_replies[:5])
        if isinstance(h, dict)
    )
    user_msg = f"""CUSTOMER MESSAGE: {customer_text}

DETECTED INTENT: {intent}
ESCALATION DECISION: {decision}

RETRIEVED AMAZON HELP REPLIES (from historical twcs.csv data):
{retrieved_str}

AI DRAFT REPLY:
{draft_reply}

Score the AI DRAFT REPLY using the rubric."""

    if OPENAI_API_KEY:
        import openai
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
        def _call():
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RUBRIC_TEXT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        return _call()

    elif GEMINI_API_KEY:
        import google.generativeai as genai
        from tenacity import retry, stop_after_attempt, wait_exponential

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
        def _call():
            resp = model.generate_content(f"{RUBRIC_TEXT}\n\nUser: {user_msg}")
            return resp.text.strip()
        return _call()

    else:
        raise RuntimeError("No LLM key set. Add OPENAI_API_KEY or GEMINI_API_KEY to .env")


def parse_scores(raw):
    """Parse JSON from judge response, with fallback for markdown-wrapped JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    return json.loads(raw)


if __name__ == "__main__":
    # Save rubric for reference
    RUBRIC_FILE.write_text(RUBRIC_TEXT, encoding="utf-8")

    if not RESULTS_JSON.exists():
        sys.exit(f"ERROR: {RESULTS_JSON} not found. Run eval/metrics.py first.")

    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)

    # Judge a subset: up to 50 examples (cost control; enough for agreement study)
    JUDGE_N = 50
    to_judge = [r for r in results if r.get("draft_reply")][:JUDGE_N]
    print(f"[INFO] Judging {len(to_judge)} examples (subset of {len(results)} total)...")

    judge_results = []
    for i, r in enumerate(to_judge):
        print(f"\r  [{i+1}/{len(to_judge)}] {r['message_id']} ...", end="", flush=True)
        try:
            raw = call_judge(
                customer_text=r["input_text"],
                retrieved_replies=r.get("retrieved_hits", []),
                draft_reply=r["draft_reply"],
                intent=r["pred_intent"],
                decision=r["pred_decision"],
            )
            scores = parse_scores(raw)
            scores["message_id"] = r["message_id"]
            scores["gold_intent"] = r["gold_intent"]
            scores["raw_response"] = raw
            judge_results.append(scores)
        except Exception as e:
            print(f"\n  [WARN] Failed to judge {r['message_id']}: {e}")
            judge_results.append({
                "message_id": r["message_id"],
                "groundedness": -1, "accuracy": -1,
                "helpfulness": -1, "tone": -1, "total": -1,
                "overall_comment": f"ERROR: {e}",
                "raw_response": "",
            })
        time.sleep(0.3)   # rate-limit buffer

    print()

    with open(JUDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Judge results saved -> {JUDGE_JSON}")

    # Print summary
    valid = [j for j in judge_results if j.get("total", -1) >= 0]
    if valid:
        avg_total = sum(j["total"] for j in valid) / len(valid)
        avg_g     = sum(j["groundedness"] for j in valid) / len(valid)
        avg_a     = sum(j["accuracy"]     for j in valid) / len(valid)
        avg_h     = sum(j["helpfulness"]  for j in valid) / len(valid)
        avg_t     = sum(j["tone"]         for j in valid) / len(valid)
        print(f"\n[JUDGE SUMMARY — {len(valid)} examples]")
        print(f"  Mean total score:    {avg_total:.2f} / 12")
        print(f"  Mean groundedness:   {avg_g:.2f} / 3")
        print(f"  Mean accuracy:       {avg_a:.2f} / 3")
        print(f"  Mean helpfulness:    {avg_h:.2f} / 3")
        print(f"  Mean tone:           {avg_t:.2f} / 3")
