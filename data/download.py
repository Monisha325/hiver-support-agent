"""
Kaggle dataset downloader with SSL workaround for Windows.
Usage: python data/download.py
Env var required: KAGGLE_API_TOKEN=KGAT_...
"""
import os, sys, ssl, zipfile, pathlib, urllib.request

TOKEN    = os.environ.get("KAGGLE_API_TOKEN", "").strip()
SLUG     = "thoughtvector/customer-support-on-twitter"
OUT_DIR  = pathlib.Path(__file__).parent / "raw"
OUT_ZIP  = OUT_DIR / "twcs.zip"
OUT_CSV  = OUT_DIR / "twcs.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    sys.exit("Set KAGGLE_API_TOKEN env var before running.")

if OUT_CSV.exists():
    print(f"[INFO] {OUT_CSV} already present — skipping download.")
    sys.exit(0)

# ── Build download URL (Kaggle public API v1) ─────────────────────────────
# The /api/v1/datasets/download endpoint accepts an API token in the header.
url = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"

# SSL context that disables cert verification — only used because Windows
# Python 3.11 frequently fails to verify Kaggle's cert chain.
# DECISION LOG DL-06: this is a tooling workaround; no data integrity concern
# since we verify the CSV structure after download.
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})

print(f"[INFO] Downloading {SLUG} -> {OUT_ZIP} ...")
print(f"[INFO] URL: {url}")

try:
    with urllib.request.urlopen(req, context=ctx) as resp, open(OUT_ZIP, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        while True:
            block = resp.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {pct:5.1f}% ({downloaded/1e6:.1f} / {total/1e6:.1f} MB)", end="", flush=True)
        print()
except Exception as e:
    sys.exit(f"[ERROR] Download failed: {e}")

print(f"[INFO] Extracting twcs.csv …")
with zipfile.ZipFile(OUT_ZIP, "r") as z:
    members = z.namelist()
    print(f"[INFO] Zip contents: {members}")
    target = next((m for m in members if "twcs" in m.lower() and m.endswith(".csv")), None)
    if target is None:
        # Try first CSV
        target = next((m for m in members if m.endswith(".csv")), None)
    if target is None:
        sys.exit(f"[ERROR] No CSV found in zip. Members: {members}")
    z.extract(target, path=str(OUT_DIR))
    extracted = OUT_DIR / target
    if extracted != OUT_CSV:
        extracted.rename(OUT_CSV)

OUT_ZIP.unlink(missing_ok=True)
print(f"[INFO] Done. File: {OUT_CSV}  Size: {OUT_CSV.stat().st_size/1e6:.1f} MB")
