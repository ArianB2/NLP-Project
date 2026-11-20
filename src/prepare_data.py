# Step 1: combine the CONDA files into one table we can use for everything.
#
# - label: E or I -> 1 (hostile), A or O -> 0. These are CONDA's labels,
#   our own hand labels come later.
# - has_profanity: 1 if the message has a word from our profanity list
# - context: the 3 lines before the message in the same match,
#   marked self / ally / enemy (CONDA's Id column is in time order)
# - split: train / dev / test (80/10/10), split by MATCH. The official train
#   and valid files share a lot of matches, so we re-split them ourselves.
#   The CONDA test file has no labels so we only use it for context.
#
# Run: python src/prepare_data.py

import numpy as np
import pandas as pd

from common import (CONTEXT_LINES, PROCESSED_CSV, RAW_DIR, clean_text,
                    format_context, has_profanity, role_of)

# load all three files
parts = []
for name in ["train", "valid", "test"]:
    part = pd.read_csv(RAW_DIR / f"CONDA_{name}.csv")
    part["source_file"] = name
    parts.append(part)
df = pd.concat(parts).sort_values("Id").reset_index(drop=True)

# text, labels, profanity
df["text"] = df["utterance"].apply(clean_text)
df["conda_intent"] = df["intentClass"].fillna("")
df["label"] = df["conda_intent"].map({"E": 1, "I": 1, "A": 0, "O": 0})
df["has_profanity"] = df["text"].apply(has_profanity).astype(int)

# context: previous 3 lines from the same match
context_of = {}
context_ids_of = {}
for match_id, match in df.groupby("matchId"):
    rows = match.to_dict("records")
    for i, row in enumerate(rows):
        earlier = rows[max(0, i - CONTEXT_LINES):i]
        lines = [(role_of(row["playerSlot"], e["playerSlot"]), e["text"]) for e in earlier]
        context_of[row["Id"]] = format_context(lines)
        context_ids_of[row["Id"]] = ";".join(str(e["Id"]) for e in earlier)
df["context"] = df["Id"].map(context_of)
df["context_ids"] = df["Id"].map(context_ids_of)

# split by match
matches = df.loc[df["label"].notna(), "matchId"].unique()
rng = np.random.default_rng(42)
matches = rng.permutation(matches)
n_train = int(len(matches) * 0.8)
n_dev = int(len(matches) * 0.1)
split_of = {}
for i, m in enumerate(matches):
    if i < n_train:
        split_of[m] = "train"
    elif i < n_train + n_dev:
        split_of[m] = "dev"
    else:
        split_of[m] = "test"
df["split"] = df["matchId"].map(split_of)
# unlabeled or empty messages are only used as context
df.loc[df["label"].isna() | (df["text"] == ""), "split"] = "context_only"

# save
df = df.rename(columns={"Id": "id", "matchId": "match_id",
                        "conversationId": "conversation_id",
                        "chatTime": "chat_time", "playerSlot": "player_slot"})
cols = ["id", "match_id", "conversation_id", "chat_time", "player_slot", "text",
        "context", "context_ids", "conda_intent", "label", "has_profanity",
        "split", "source_file"]
PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
df[cols].to_csv(PROCESSED_CSV, index=False)
print("saved", len(df), "rows to", PROCESSED_CSV)

# summary for the paper
used = df[df["split"] != "context_only"]
print(used.groupby("split").agg(messages=("id", "size"),
                                matches=("match_id", "nunique"),
                                hostile_rate=("label", "mean"),
                                profanity_rate=("has_profanity", "mean")).round(3))
print("\nprofanity rate by CONDA label:")
print(used.groupby("conda_intent")["has_profanity"].mean().round(3))
