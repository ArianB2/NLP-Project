"""Sanity checks on the processed data. These are the three checks promised
in our plan, plus a couple of cheap extras.

Run:  python -m pytest tests/      (or:  python tests/test_data.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import GOLD_CSV, has_profanity, load_processed  # noqa: E402

DF = load_processed()
USED = DF[DF["split"].isin(["train", "dev", "test"])]


def test_no_match_in_two_splits():
    # Each match must belong to exactly one of train / dev / test.
    splits_per_match = USED.groupby("match_id")["split"].nunique()
    assert splits_per_match.max() == 1


def test_no_conversation_in_two_splits():
    splits_per_conv = USED.groupby("conversation_id")["split"].nunique()
    assert splits_per_conv.max() == 1


def test_labels_are_binary():
    assert set(USED["label"].unique()) <= {0, 1}


def test_context_only_from_earlier_lines_in_same_match():
    match_of = dict(zip(DF["id"], DF["match_id"]))
    for row in USED.itertuples():
        if not row.context_ids:
            continue
        for cid in map(int, row.context_ids.split(";")):
            assert cid < row.id, f"context {cid} is after message {row.id}"
            assert match_of[cid] == row.match_id, "context from another match"


def test_context_has_at_most_3_lines():
    n_lines = USED["context_ids"].map(lambda s: len(s.split(";")) if s else 0)
    assert n_lines.max() <= 3


def test_profanity_flag_examples():
    assert has_profanity("send the damn courier")
    assert has_profanity("WTF that tide")
    assert not has_profanity("pass the ball")      # "ass" inside a word
    assert not has_profanity("noob mid report")    # insult, not profanity


def test_gold_labels_are_from_test_matches():
    # Hand labels must come only from held-out test matches.
    if not GOLD_CSV.exists():
        return
    import pandas as pd
    gold = pd.read_csv(GOLD_CSV)
    split_of = dict(zip(DF["id"], DF["split"]))
    assert all(split_of[i] == "test" for i in gold["id"])
    assert set(gold["gold_label"].unique()) <= {0, 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
