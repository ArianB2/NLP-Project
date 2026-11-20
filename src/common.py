# Helper functions and file paths shared by all the other scripts.

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# folders
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_CSV = ROOT / "data" / "processed" / "conda_processed.csv"
LABELS_DIR = ROOT / "data" / "labels"
GOLD_CSV = LABELS_DIR / "gold.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"

CONTEXT_LINES = 3  # how many earlier chat lines the context model gets


# ---- profanity flag ----
# read the word patterns from resources/profanity_patterns.txt
patterns = []
for line in (ROOT / "resources" / "profanity_patterns.txt").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        patterns.append(line)
# \b means whole word only, so "ass" doesn't match "pass"
PROFANITY_RE = re.compile(r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE)


def has_profanity(text):
    return PROFANITY_RE.search(text or "") is not None


# ---- text and context ----
def clean_text(text):
    # CONDA puts [SEPA] between lines from the same player. We swap it for " | ".
    if not isinstance(text, str):
        return ""
    text = text.replace("[SEPA]", " | ")
    return " ".join(text.split())


def role_of(speaker_slot, other_slot):
    # In Dota 2, slots 0-4 are one team and 5-9 are the other team.
    if other_slot == speaker_slot:
        return "self"
    if (speaker_slot < 5) == (other_slot < 5):
        return "ally"
    return "enemy"


def format_context(lines):
    # lines is a list of (role, text), oldest first
    # result looks like: "ally: mid noob || self: shut up"
    return " || ".join(role + ": " + text for role, text in lines)


def load_processed():
    if not PROCESSED_CSV.exists():
        raise SystemExit("Run python src/prepare_data.py first")
    df = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


# ---- metrics ----
def hostile_metrics(y_true, y_pred):
    # precision / recall / F1 for the hostile class (label 1) only
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "precision": round(float(p[0]), 3),
        "recall": round(float(r[0]), 3),
        "f1": round(float(f[0]), 3),
        "accuracy": round(float((tp + tn) / len(y_true)), 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
