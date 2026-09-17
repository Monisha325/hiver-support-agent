"""Read and print all golden set rows for labelling review."""
import csv
with open("eval/golden_set.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Total rows: {len(rows)}")
for r in rows:
    mid   = r["message_id"]
    batch = r["sampling_batch"][:10]
    intent = r["gold_intent"][:30]
    text   = r["input_text"][:140]
    print(f"[{mid}|{batch}] {intent:30s} | {text}")
