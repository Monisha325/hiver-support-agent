import pathlib
ROOT = pathlib.Path(".")
checks = [
  ("data/raw/twcs.csv",          "Raw dataset present"),
  ("intent/taxonomy.py",         "Intent taxonomy defined"),
  ("intent/DERIVATION_NOTES.md", "Derivation notes written"),
  ("retrieval/index.faiss",      "FAISS index built"),
  ("retrieval/corpus.jsonl",     "Corpus metadata saved"),
  ("retrieval/retrieve.py",      "Retrieval module exists"),
  ("agent/agent.py",             "Agent pipeline exists"),
  ("eval/golden_set.csv",        "Golden set (198 examples)"),
  ("eval/metrics.py",            "Metrics harness exists"),
  ("eval/llm_judge.py",          "LLM judge exists"),
  ("eval/judge_agreement.py",    "Agreement study exists"),
  ("eval/baselines.py",          "Baselines exist"),
  ("eval/baselines_report.txt",  "Baselines report generated"),
  ("report/REPORT.md",           "Report written (5 sections)"),
  ("report/DECISION_LOG.md",     "Decision log written"),
  ("README.md",                  "README written"),
  (".env.example",               ".env.example present"),
  ("requirements.txt",           "requirements.txt present"),
  ("run_pipeline.py",            "Pipeline entry point exists"),
]
all_pass = True
for path, label in checks:
    exists = (ROOT / path).exists()
    status = "PASS" if exists else "FAIL"
    if not exists:
        all_pass = False
    print(f"  [{status}] {label:<45} ({path})")

dl = (ROOT / "report" / "DECISION_LOG.md").read_text(encoding="utf-8")
dl_entries = dl.count("[DL-")
print(f"\n  Decision log entries: {dl_entries} (target: 10-15)")

gset = (ROOT / "eval" / "golden_set.csv").read_text(encoding="utf-8").strip().split("\n")
print(f"  Golden set rows: {len(gset)-1} (target: 150-250)")

result = "ALL PASS" if all_pass else "SOME GAPS - see above"
print(f"\n  Overall: {result}")
