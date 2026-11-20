# Systems 3 and 4: fine-tune DistilBERT.
#
#   python src/train_distilbert.py             -> message only
#   python src/train_distilbert.py --context   -> 3 earlier lines + message
#
# Same model, data and settings for both. Only the input changes, so any
# difference in results comes from the context.
#
# With context we give the tokenizer two texts, and it builds:
#   [CLS] ally: go mid || enemy: ez [SEP] mid noob [SEP]
# If it's too long we cut from the left, so the oldest context goes first.
#
# Training: for each batch -> tokenize -> model -> loss -> backward -> update.
# After every epoch we check F1 on the dev set and keep the best epoch.
#
# Use a Colab GPU for this (takes ~10-20 min per model on a T4).

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import MODELS_DIR, PRED_DIR, hostile_metrics, load_processed

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2).to(device)
    return tokenizer, model


def encode(tokenizer, texts, contexts, use_context):
    if use_context:
        return tokenizer(list(contexts), list(texts), padding=True, truncation=True,
                         max_length=128, return_tensors="pt").to(device)
    return tokenizer(list(texts), padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(device)


def predict_scores(tokenizer, model, texts, contexts, use_context):
    # returns the probability of "hostile" for each message
    model.eval()
    texts, contexts = list(texts), list(contexts)
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            inputs = encode(tokenizer, texts[i:i + 128], contexts[i:i + 128], use_context)
            probs = torch.softmax(model(**inputs).logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return np.array(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=0, help="only use N rows (quick test)")
    args = parser.parse_args()

    torch.manual_seed(42)
    df = load_processed()
    train = df[df["split"] == "train"].reset_index(drop=True)
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        train, dev, test = train.head(args.limit), dev.head(args.limit), test.head(args.limit)

    name = "distilbert_ctx" if args.context else "distilbert_msg"
    save_dir = MODELS_DIR / name
    tokenizer, model = load_model(args.model)
    print("device:", device, "| context:", args.context, "| train rows:", len(train))

    batch_size = 32
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # learning rate goes up for the first 10% of steps, then down to 0
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1 = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train.sample(frac=1, random_state=epoch)
        for step in range(steps_per_epoch):
            batch = shuffled.iloc[step * batch_size:(step + 1) * batch_size]
            inputs = encode(tokenizer, batch["text"], batch["context"], args.context)
            labels = torch.tensor(batch["label"].astype(int).values).to(device)

            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}")

        dev_scores = predict_scores(tokenizer, model, dev["text"], dev["context"], args.context)
        m = hostile_metrics(dev["label"].astype(int), (dev_scores >= 0.5).astype(int))
        print(f"epoch {epoch} dev: precision {m['precision']} recall {m['recall']} f1 {m['f1']}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print("saved best model to", save_dir)

    # load the best epoch and predict the test set
    tokenizer, model = load_model(save_dir)
    scores = predict_scores(tokenizer, model, test["text"], test["context"], args.context)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "score": scores.round(4),
                  "pred": (scores >= 0.5).astype(int)}).to_csv(PRED_DIR / f"{name}.csv", index=False)
    print("best dev f1:", best_f1, "| test predictions saved")
