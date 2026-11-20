# System 2: unitary/toxic-bert, zero-shot (we don't train it).
#
# toxic-bert was trained on Wikipedia comments. It outputs 6 scores from 0 to 1:
#   toxic, severe_toxic, obscene, threat, insult, identity_hate
# Each one uses its own sigmoid (multi-label), so they don't add up to 1.
#
# We turn them into 0/1 with two rules we picked before seeing results:
#   Rule A: toxic >= 0.5
#   Rule B: insult >= 0.5 or threat >= 0.5   (ignores "obscene" = profanity)
#
# Run: python src/toxicbert_baseline.py

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import PRED_DIR, RESULTS_DIR, load_processed

MODEL_NAME = "unitary/toxic-bert"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_toxicbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


def toxicbert_scores(texts, tokenizer, model):
    # returns a table with one column per toxic-bert label
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = list(texts[i:i + 64])
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            all_scores.append(torch.sigmoid(logits).cpu())
    return pd.DataFrame(torch.cat(all_scores).numpy(), columns=labels)


if __name__ == "__main__":
    df = load_processed()
    test = df[df["split"] == "test"].reset_index(drop=True)
    print("scoring", len(test), "test messages...")

    tokenizer, model = load_toxicbert()
    scores = toxicbert_scores(test["text"], tokenizer, model)
    scores.insert(0, "id", test["id"])
    scores.round(4).to_csv(RESULTS_DIR / "toxicbert_scores.csv", index=False)

    rule_a = scores["toxic"]
    rule_b = scores[["insult", "threat"]].max(axis=1)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in [("A", rule_a), ("B", rule_b)]:
        pd.DataFrame({"id": test["id"], "score": s.round(4),
                      "pred": (s >= 0.5).astype(int)}).to_csv(
            PRED_DIR / f"toxicbert_{name}.csv", index=False)
        print(f"rule {name}: flagged {(s >= 0.5).mean():.1%} of test messages")
