"""Print real failure examples from agent_results.json for REPORT.md."""
import json
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
results_path = ROOT / "eval" / "agent_results.json"
golden_path  = ROOT / "eval" / "golden_set.csv"

if not results_path.exists():
    sys.exit("ERROR: eval/agent_results.json not found.")

with open(results_path, encoding="utf-8") as f:
    results = json.load(f)
with open(golden_path, encoding="utf-8") as f:
    golden = {r["message_id"]: r for r in csv.DictReader(f)}

print("=== FALSE-AUTO-HANDLE (gold=escalate, pred=auto) — MISSED ESCALATIONS ===")
false_auto = [r for r in results if r["gold_decision"] == "escalate" and r["pred_decision"] == "auto"]
for r in false_auto:
    print(f"\n[{r['message_id']}] gold_intent={r['gold_intent']}  pred_intent={r['pred_intent']}")
    print(f"  TEXT:  {r['input_text'][:200]}")
    print(f"  DRAFT: {r['draft_reply'][:150]}")

print("\n\n=== FALSE-ESCALATE (gold=auto, pred=escalate) — UNNECESSARY ROUTING ===")
false_esc = [r for r in results if r["gold_decision"] == "auto" and r["pred_decision"] == "escalate"]
for r in false_esc[:5]:
    print(f"\n[{r['message_id']}] gold_intent={r['gold_intent']}  pred_intent={r['pred_intent']}")
    print(f"  TEXT: {r['input_text'][:200]}")

print(f"\n\nTotal false-auto: {len(false_auto)}, false-escalate: {len(false_esc)}")
