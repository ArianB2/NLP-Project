# Step 2a: make the labeling sheets for Arian and Wadood.
#
# We pick 300 messages from the TEST matches only (so they never show up in
# training). We pick more abuse than normal chat would have, so we get enough
# of the cases we care about:
#   100 E, 60 I, 30 A, 50 O with profanity, 60 O without profanity
# (Remember to say in the paper that this sample isn't like real chat.)
#
# The first 50 go in both sheets so we can check if we agree.
# The CONDA label is hidden from the sheets so it doesn't affect us.
#
# Run: python src/make_label_sheet.py

import pandas as pd

from common import LABELS_DIR, load_processed

df = load_processed()
test = df[df["split"] == "test"].copy()
test["lower"] = test["text"].str.lower()
test = test.drop_duplicates("lower")  # don't label "noob" 20 times

groups = [
    ("E", test[test["conda_intent"] == "E"], 100),
    ("I", test[test["conda_intent"] == "I"], 60),
    ("A", test[test["conda_intent"] == "A"], 30),
    ("O_profanity", test[(test["conda_intent"] == "O") & (test["has_profanity"] == 1)], 50),
    ("O_clean", test[(test["conda_intent"] == "O") & (test["has_profanity"] == 0)], 60),
]
picked = []
for name, pool, n in groups:
    sample = pool.sample(n=min(n, len(pool)), random_state=42)
    sample["group"] = name
    picked.append(sample)
sample = pd.concat(picked).sample(frac=1, random_state=42)  # shuffle

shared = sample.iloc[:50]
rest = sample.iloc[50:]
half = len(rest) // 2
sheets = {"arian": pd.concat([shared, rest.iloc[:half]]),
          "wadood": pd.concat([shared, rest.iloc[half:]])}

LABELS_DIR.mkdir(parents=True, exist_ok=True)
for person, sheet in sheets.items():
    out = pd.DataFrame({
        "id": sheet["id"],
        "context (earlier lines, oldest first)": sheet["context"],
        "message": sheet["text"],
        "label (1 = hostile toward a player, 0 = not)": "",
        "unsure (x)": "",
        "notes": "",
    })
    out = out.sample(frac=1, random_state=42)  # mix the shared rows in
    out.to_csv(LABELS_DIR / f"label_sheet_{person}.csv", index=False)

# answer key, don't open this while labeling
key = sample[["id", "group", "conda_intent", "has_profanity"]].copy()
key["shared"] = key["id"].isin(shared["id"]).astype(int)
key.to_csv(LABELS_DIR / "sample_key.csv", index=False)

print(len(sample), "messages picked, 50 shared")
print(key["group"].value_counts())
