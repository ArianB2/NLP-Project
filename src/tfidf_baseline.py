# System 1: TF-IDF + logistic regression (message only).
#
# TF-IDF gives every word a score: high if it's common in this message but
# rare in the whole dataset. Logistic regression learns one weight per word.
# A big positive weight means the word pushes toward "hostile".
# Since it's one weight per word, we can look at the top words and see if
# they're mostly swear words.
#
# Run: python src/tfidf_baseline.py

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import (FIG_DIR, MODELS_DIR, PRED_DIR, RESULTS_DIR, has_profanity,
                    hostile_metrics, load_processed)

df = load_processed()
train = df[df["split"] == "train"]
dev = df[df["split"] == "dev"]
test = df[df["split"] == "test"]

# words and 2-word phrases; the token pattern keeps 1-letter words like "u"
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w+\b"),
    LogisticRegression(max_iter=2000),
)
model.fit(train["text"], train["label"])
print("dev:", hostile_metrics(dev["label"], model.predict(dev["text"])))

# save test predictions and the model
scores = model.predict_proba(test["text"])[:, 1]
PRED_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": test["id"], "score": scores.round(4),
              "pred": (scores >= 0.5).astype(int)}).to_csv(PRED_DIR / "tfidf_lr.csv", index=False)
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "tfidf_lr.joblib")

# top 30 words pushing toward hostile
words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
weights = model.named_steps["logisticregression"].coef_[0]
top = pd.DataFrame({"feature": words, "weight": weights.round(3)})
top = top.sort_values("weight", ascending=False).head(30)
top["is_profanity"] = top["feature"].apply(has_profanity).astype(int)
top.to_csv(RESULTS_DIR / "tfidf_top_features.csv", index=False)
share = top["is_profanity"].mean()
print(f"\ntop 30 hostile features ({share:.0%} are profanity):")
print(top.to_string(index=False))

# bar chart, red = profanity
top = top.iloc[::-1]
colors = ["#c0392b" if p else "#7f8c8d" for p in top["is_profanity"]]
plt.figure(figsize=(6, 8))
plt.barh(top["feature"], top["weight"], color=colors)
plt.xlabel("weight toward hostile")
plt.title(f"TF-IDF + LR: top 30 hostile features\nred = profanity ({share:.0%})")
plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "tfidf_top_features.png", dpi=200)
