"""
Quick inspection of UNCLASSIFIED tweets to find hidden intents.
Run after derive_taxonomy.py.
"""
import pathlib, re, sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT       = pathlib.Path(__file__).parent.parent
INTENT_DIR = ROOT / "intent"
RAW_CSV    = ROOT / "data" / "raw" / "twcs.csv"
BRAND_ID   = "AmazonHelp"
SEED       = 42

INTENT_SEEDS = {
    "ORDER_DELIVERY_STATUS": ["where","order","package","tracking","delivery","delivered","arrive","arrived","expected","ship","shipping","shipment","status","track","dispatched","carrier","usps","ups","fedex","estimated","arrival","late","delayed","delay","not received","missing package"],
    "RETURN_REFUND_REPLACEMENT": ["return","refund","replacement","exchange","money back","reimburse","reimbursement","credit","sent back","sending back","damaged","broken","defective","wrong item","incorrect","not as described","replace","send another"],
    "ACCOUNT_LOGIN_ACCESS": ["account","login","sign in","password","locked","access","verify","verification","email","2fa","two factor","suspended","banned","hacked","unauthorized","forgot password","reset","cannot log","can't log"],
    "ORDER_CANCELLATION": ["cancel","cancelled","cancellation","cancel order","stop order","don't want","do not want","withdraw"],
    "CHARGE_PAYMENT_BILLING": ["charge","charged","billing","bill","payment","paid","double charged","overcharged","invoice","receipt","card","credit card","debit","unauthorized charge","prime charge","subscription","fee","cost"],
    "PRIME_SUBSCRIPTION": ["prime","prime membership","prime video","prime day","prime subscription","free trial","membership","annual","student prime","cancel prime","prime benefits"],
    "PRODUCT_QUALITY_COMPLAINT": ["quality","fake","counterfeit","not working","doesn't work","broken","poor quality","disappointed","terrible","awful","complaint","issue with","problem with","defective product","used","opened","expired","wrong","not genuine"],
    "SELLER_THIRD_PARTY": ["seller","third party","marketplace","vendor","sold by","fulfilled","independent seller","merchant","fraudulent seller","scam seller"],
    "DELIVERY_ADDRESS_CHANGE": ["address","change address","wrong address","update address","redirect","delivery address","shipping address"],
    "GENERAL_INQUIRY_OTHER": ["help","question","wondering","how do","can you","what is","please","information","assist"],
}

def clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"@\w+","",text)
    text = re.sub(r"#\w+","",text)
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def seed_score(text, kws):
    return sum(1 for kw in kws if kw in text)

print("[INFO] Loading twcs.csv ...")
df = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
df["inbound"] = df["inbound"].str.strip().str.lower().map({"true":True,"false":False})
brand_out = df[(df["author_id"]==BRAND_ID)&(df["inbound"]==False)]
replied_ids = set(brand_out["in_response_to_tweet_id"].dropna().unique())
inbound = df[(df["inbound"]==True)&(df["tweet_id"].isin(replied_ids))].copy()
sample = inbound.sample(n=min(50000,len(inbound)), random_state=SEED).copy()
sample["clean_text"] = sample["text"].apply(clean)

for intent, kws in INTENT_SEEDS.items():
    sample[f"score_{intent}"] = sample["clean_text"].apply(lambda t: seed_score(t, kws))

score_cols = [f"score_{k}" for k in INTENT_SEEDS]
sample["best_intent"] = sample[score_cols].idxmax(axis=1).str.replace("score_","")
sample["best_score"]  = sample[score_cols].max(axis=1)
sample.loc[sample["best_score"]==0,"best_intent"] = "UNCLASSIFIED"

unclassified = sample[sample["best_intent"]=="UNCLASSIFIED"]["clean_text"].tolist()
print(f"\n[INFO] Unclassified: {len(unclassified):,} tweets")

# TF-IDF on unclassified to find hidden topics
print("\n[TOP 60 TERMS IN UNCLASSIFIED BUCKET]")
cv = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words="english", min_df=3, max_df=0.7)
Xu = cv.fit_transform(unclassified)
terms = cv.get_feature_names_out()
scores = np.asarray(Xu.mean(axis=0)).flatten()
top = sorted(zip(terms, scores), key=lambda x: -x[1])[:60]
for t, s in top:
    print(f"  {t:<35} {s:.5f}")

# Print 15 random unclassified examples
print("\n[15 RANDOM UNCLASSIFIED EXAMPLES]")
unc_df = sample[sample["best_intent"]=="UNCLASSIFIED"]
for _, row in unc_df.sample(15, random_state=99).iterrows():
    print(f"  - {row['text'][:180]}")
