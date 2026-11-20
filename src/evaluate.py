# Step 4: score every system that has a predictions file.
#
# Two test sets:
#   conda_test = all test-match messages, with CONDA's labels (E, I -> 1)
#   gold       = our ~300 hand labels (hostile toward a player). Main result.
#
# For each system we get precision / recall / F1 for the hostile class,
# split into messages with and without profanity. We also count false
# positives on profanity that isn't aimed at a player (the "stupid valve" case).
#
# Also saves a confusion matrix per system, a bar chart, and 20 mistakes per
# system in results/errors/ so we can tag them (sarcasm, banter,
# game-directed profanity, implicit hostility, other).
#
# Run: python src/evaluate.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from common import FIG_DIR, GOLD_CSV, PRED_DIR, RESULTS_DIR, hostile_metrics, load_processed

SYSTEMS = {
    "tfidf_lr": "TF-IDF + LR",
    "toxicbert_A": "toxic-bert (Rule A)",
    "toxicbert_B": "toxic-bert (Rule B)",
    "distilbert_msg": "DistilBERT (message)",
    "distilbert_ctx": "DistilBERT (context)",
}

df = load_processed()
test = df[df["split"] == "test"].copy()
test["true"] = test["label"].astype(int)
test_sets = {"conda_test": test}
if GOLD_CSV.exists():
    gold = test.merge(pd.read_csv(GOLD_CSV), on="id")
    gold["true"] = gold["gold_label"]
    test_sets["gold"] = gold
else:
    print("no gold.csv yet, only using CONDA labels")
main_set = "gold" if "gold" in test_sets else "conda_test"

results = []
slices = []
FIG_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "errors").mkdir(parents=True, exist_ok=True)

for key, name in SYSTEMS.items():
    path = PRED_DIR / f"{key}.csv"
    if not path.exists():
        print("skipping", name, "(no predictions yet)")
        continue
    preds = pd.read_csv(path)

    for set_name, data in test_sets.items():
        d = data.merge(preds, on="id")
        row = {"system": name, "test_set": set_name}
        row.update(hostile_metrics(d["true"], d["pred"]))
        benign = d[(d["has_profanity"] == 1) & (d["true"] == 0)]
        row["fp_on_benign_profanity"] = f"{benign['pred'].sum()}/{len(benign)}"
        results.append(row)

        for flag, slice_name in [(1, "with profanity"), (0, "no profanity")]:
            s = d[d["has_profanity"] == flag]
            m = hostile_metrics(s["true"], s["pred"])
            slices.append({"system": name, "test_set": set_name, "slice": slice_name,
                           "n": m["n"], "precision": m["precision"],
                           "recall": m["recall"], "f1": m["f1"]})

        if set_name == main_set:
            # confusion matrix
            ConfusionMatrixDisplay.from_predictions(
                d["true"], d["pred"], display_labels=["not hostile", "hostile"],
                cmap="Blues", colorbar=False)
            plt.title(f"{name} ({set_name})", fontsize=9)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"confusion_{key}.png", dpi=200)
            plt.close()

            # 20 random mistakes to look at by hand
            wrong = d[d["pred"] != d["true"]]
            wrong = wrong.sample(n=min(20, len(wrong)), random_state=42)
            wrong = wrong[["id", "context", "text", "true", "pred", "score",
                           "has_profanity", "conda_intent"]]
            wrong["error_type"] = ""
            wrong.to_csv(RESULTS_DIR / "errors" / f"{key}.csv", index=False)

results = pd.DataFrame(results)
slices = pd.DataFrame(slices)
results.to_csv(RESULTS_DIR / "results_table.csv", index=False)
slices.to_csv(RESULTS_DIR / "profanity_slices.csv", index=False)
cols = ["system", "test_set", "n", "precision", "recall", "f1", "fp_on_benign_profanity"]
print(results[cols].to_string(index=False))
print()
main_slices = slices[slices["test_set"] == main_set]
print(main_slices.to_string(index=False))

# bar chart: F1 with vs without profanity
table = main_slices.pivot(index="system", columns="slice", values="f1")
table = table.reindex([n for n in SYSTEMS.values() if n in table.index])
table[["with profanity", "no profanity"]].plot.bar(color=["#c0392b", "#2980b9"], figsize=(8, 4))
plt.ylabel("F1 (hostile class)")
plt.ylim(0, 1)
plt.title(f"F1 with vs without profanity ({main_set})")
plt.xticks(rotation=20, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "f1_by_profanity.png", dpi=200)
