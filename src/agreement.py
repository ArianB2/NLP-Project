# Step 2b: check how much we agree, then make gold.csv (our final labels).
#
# Before running: download both Google Sheets as CSV and save them over
# data/labels/label_sheet_arian.csv and data/labels/label_sheet_wadood.csv
#
# 1. Cohen's kappa on the 50 shared rows. It's agreement after removing the
#    agreement you'd get by chance. 1 = perfect, 0 = random. Above 0.6 is good.
# 2. Disagreements go to data/labels/disagreements.csv. We talk about each
#    one, write the answer in final_label, and run this again.
# 3. gold.csv = everything we agree on + resolved ones + each person's own rows.
#
# Run: python src/agreement.py

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import GOLD_CSV, LABELS_DIR

LABEL_COL = "label (1 = hostile toward a player, 0 = not)"


def read_sheet(person):
    sheet = pd.read_csv(LABELS_DIR / f"label_sheet_{person}.csv")
    sheet = sheet.rename(columns={LABEL_COL: "label"})
    sheet = sheet[sheet["label"].isin([0, 1])]  # skip blanks
    print(person, "labeled", len(sheet), "rows")
    return sheet[["id", "message", "label"]]


arian = read_sheet("arian")
wadood = read_sheet("wadood")
key = pd.read_csv(LABELS_DIR / "sample_key.csv")
shared_ids = set(key.loc[key["shared"] == 1, "id"])

# 1. agreement on the shared rows
both = arian[arian["id"].isin(shared_ids)].merge(
    wadood[["id", "label"]], on="id", suffixes=("_arian", "_wadood"))
kappa = cohen_kappa_score(both["label_arian"], both["label_wadood"])
agree = (both["label_arian"] == both["label_wadood"]).mean()
print(f"\nshared rows: {len(both)}  agreement: {agree:.1%}  kappa: {kappa:.3f}")

# 2. disagreements (keep any final_label we already filled in)
dis_path = LABELS_DIR / "disagreements.csv"
resolved = {}
if dis_path.exists():
    old = pd.read_csv(dis_path)
    old = old[old["final_label"].isin([0, 1])]
    resolved = dict(zip(old["id"], old["final_label"].astype(int)))
dis = both[both["label_arian"] != both["label_wadood"]].copy()
dis["final_label"] = dis["id"].map(resolved)
dis.to_csv(dis_path, index=False)
print(f"disagreements: {len(dis)}, resolved: {dis['final_label'].notna().sum()}")

# 3. gold labels
gold = []
for _, row in both.iterrows():
    if row["label_arian"] == row["label_wadood"]:
        gold.append((row["id"], int(row["label_arian"]), "both"))
    elif row["id"] in resolved:
        gold.append((row["id"], resolved[row["id"]], "resolved"))
for person, sheet in [("arian", arian), ("wadood", wadood)]:
    for _, row in sheet[~sheet["id"].isin(shared_ids)].iterrows():
        gold.append((row["id"], int(row["label"]), person))

gold = pd.DataFrame(gold, columns=["id", "gold_label", "labeled_by"])
gold.to_csv(GOLD_CSV, index=False)
print(f"\ngold.csv: {len(gold)} rows, {gold['gold_label'].mean():.1%} hostile")
