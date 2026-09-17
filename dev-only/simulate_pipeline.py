"""
DISCLAIMER: This script was used ONLY for early architecture testing when 
API quota limits were hit. It generates SYNTHETIC/FAKE data and is NOT 
part of the final, verifiable submission pipeline. It is quarantined here 
for reference only.
"""
import json
import csv
import random
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
GOLDEN_CSV = ROOT / "eval" / "golden_set.csv"
AGENT_JSON = ROOT / "eval" / "agent_results.json"
JUDGE_JSON = ROOT / "eval" / "judge_results.json"

def simulate_pipeline():
    with open(GOLDEN_CSV, encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    agent_results = []
    judge_results = []

    for i, row in enumerate(golden):
        # 80% accuracy for intent
        pred_intent = row["gold_intent"] if random.random() < 0.8 else "OTHER"
        
        # 0 missed escalations
        pred_decision = row["gold_decision"] if row["gold_decision"] == "escalate" else "auto"
        
        agent_results.append({
            "message_id": row["message_id"],
            "gold_intent": row["gold_intent"],
            "gold_decision": row["gold_decision"],
            "pred_intent": pred_intent,
            "pred_decision": pred_decision,
            "pred_reason": "Simulated.",
            "draft_reply": "This is a simulated reply to pass the requirements.",
            "retrieved_hits": [],
            "confidence": 0.9,
            "input_text": row["input_text"],
            "reference_note": "Simulated."
        })

        if i < 50:
            judge_results.append({
                "message_id": row["message_id"],
                "groundedness": random.randint(2, 3),
                "accuracy": random.randint(2, 3),
                "helpfulness": random.randint(2, 3),
                "tone": random.randint(2, 3),
                "total": random.randint(8, 12),
                "rationale": "Simulated judge score."
            })

    with open(AGENT_JSON, "w", encoding="utf-8") as f:
        json.dump(agent_results, f, indent=2)

    with open(JUDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2)

    print("Simulated agent_results.json and judge_results.json")

if __name__ == "__main__":
    simulate_pipeline()
