"""Explore the local training CSV: env/bin/python analyze_conda.py."""

import argparse
import os
from pathlib import Path

# Keep plotting headless and its cache within the writable project directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTENTS = {"O": "Other", "E": "Explicit toxicity", "I": "Implicit toxicity", "A": "Action"}
SLOTS = {"O": "Other", "P": "Pronoun", "S": "Game slang", "C": "Character/hero", "T": "Toxicity", "D": "Dota-specific"}
PAPER = "https://arxiv.org/html/2106.06213v2"


def table(frame):
    """Small Markdown table without an extra tabulate dependency."""
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def analyze(csv_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    # Preserve literal strings such as 'NA', which may be genuine chat messages.
    df = pd.read_csv(csv_path, keep_default_na=False)
    labels = df.slotClasses.str.split()
    counts = df.intentClass.value_counts().reindex(INTENTS, fill_value=0)
    intent_summary = pd.DataFrame({
        "Label": counts.index, "Meaning": [INTENTS[k] for k in counts.index],
        "Messages": counts.values, "Percent": (counts.values / len(df) * 100).round(2),
    })
    all_slots = labels.explode().dropna().value_counts()
    slots = all_slots.reindex(SLOTS, fill_value=0)
    slot_summary = pd.DataFrame({
        "Label": slots.index, "Meaning": [SLOTS[k] for k in slots.index],
        "Tokens": slots.values, "Percent (excludes SEPA)": (slots.values / slots.sum() * 100).round(2),
    })
    lengths = labels.map(lambda items: sum(tag != "SEPA" for tag in items))
    has_t = labels.map(lambda items: "T" in items)
    cross = pd.crosstab(df.intentClass, has_t).reindex(index=INTENTS, columns=[False, True], fill_value=0)
    cross.columns = ["No T token", "Has T token"]
    cross["Percent with T token"] = (cross["Has T token"] / cross.sum(axis=1) * 100).round(2)
    cross.index.name = "Intent"
    blanks = df.apply(lambda col: col.astype(str).str.strip().eq("").sum())
    conversations = df.groupby(["matchId", "conversationId"]).size()
    toxic = int(df.intentClass.isin(["E", "I"]).sum())
    duplicate_messages = int(df.utterance.duplicated().sum())
    unexpected = sorted(set(all_slots.index) - set(SLOTS) - {"SEPA"})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle(f"CONDA training split — {len(df):,} messages", fontsize=18)
    ax = axes[0, 0]
    bars = ax.bar([f"{k}: {v}" for k, v in INTENTS.items()], counts.values, color=["#5687a6", "#bd5050", "#d89b45", "#659c7d"])
    ax.bar_label(bars, labels=[f"{n:,}\n{n/len(df):.1%}" for n in counts.values], padding=3)
    ax.set(title="Whole-message intent: one label per row", ylabel="Messages", ylim=(0, counts.max() * 1.22))
    ax.tick_params(axis="x", labelrotation=12)
    ax = axes[0, 1]
    ax.bar(slots.index, slots.values, color="#5687a6")
    ax.set(title="Token labels (SEPA separators excluded)", ylabel="Tokens", xlabel="O other · P pronoun · S slang · C hero · T toxicity · D Dota")
    ax = axes[1, 0]
    for key in INTENTS:
        values = lengths[df.intentClass.eq(key)]
        # Percentiles summarize the full distribution; chart shows lengths <=30.
        ax.hist(values, bins=np.arange(-0.5, 31.5, 1), histtype="step", linewidth=1.8, label=key)
    ax.set(title="Message length after provided token processing", xlabel="Tokens per message, excluding SEPA (0–30 shown)", ylabel="Messages")
    ax.legend(title="Intent")
    ax = axes[1, 1]
    bars = ax.bar(cross.index, cross["Percent with T token"], color="#bd5050")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(title="Messages containing a T-tagged token, by intent", xlabel="Whole-message intent", ylabel="Percent of messages in class", ylim=(0, 110))
    fig.savefig(output / "overview.png", dpi=180)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)

    intent_summary.to_csv(output / "intent_counts.csv", index=False)
    slot_summary.to_csv(output / "slot_counts.csv", index=False)
    cross.to_csv(output / "intent_vs_toxic_tokens.csv")
    blank_table = pd.DataFrame({"Column": blanks.index, "Blank rows": blanks.values})
    report = f"""# Understanding the local CONDA training data

Generated from `{csv_path.name}` by `analyze_conda.py`. These figures describe this file only.

## What a row represents

One row is a chat utterance. The paper merges consecutive messages from one player with `[SEPA]`.
It provides two tasks: human-annotated whole-message intent and lexicon-derived token labels.
The paper describes 44,869 utterances overall, split into 26,921 training and 8,974 each for validation/test.
Only the training CSV is present locally. [Paper, sections 3 and 5.1]({PAPER}).

## Local size and intent distribution

- Messages: **{len(df):,}**; columns: **{len(df.columns)}**.
- Matches represented: **{df.matchId.nunique():,}**; match/conversation pairs: **{len(conversations):,}**.
- E + I messages: **{toxic:,} ({toxic / len(df):.2%})**.
- Always guessing O would score **{counts['O'] / len(df):.2%} accuracy** on this file. That illustrates why accuracy alone is misleading.

{table(intent_summary)}

E describes overt toxicity; I describes contextual or hidden toxicity. A covers non-toxic actions such as reporting or pausing; O is the remaining category.
These are the dataset's annotation categories, not labels inferred by this script. [Annotation instructions]({PAPER}#S3.SS3).

## Columns

| Column | How to read it |
| --- | --- |
| Id | Utterance identifier; use for tracing rows |
| matchId | Match grouping key |
| conversationId | Conversation grouping key; combine with matchId |
| utterance | Original message text, sometimes with [SEPA] |
| chatTime | In-match time; sort within a match to inspect context |
| playerSlot | Player position/slot in the match |
| playerId | Player identifier/display name in the data; avoid treating it as a globally reliable person ID |
| intentClass | One message-level class: E, I, A, O |
| slotClasses | Space-separated labels for the processed tokens |
| slotTokens | Processed tokens shown alongside their labels |

## Token labels

{table(slot_summary)}

`C` means a game character/hero, not a character in a string. `O` means other, not numeric zero.
`SEPA` appears **{int(all_slots.get('SEPA', 0)):,}** times as a separator label and is excluded from the six semantic classes above.
The supplied tokens have separate preprocessing, so splitting raw `utterance` on spaces is not a reliable way to align them with `slotClasses`.
The paper describes lexicon matching for slots, rather than manual contextual judgment for every token. [Paper, section 3.3]({PAPER}#S3.SS3).

## Why message intent and toxic words differ

{table(cross.reset_index())}

The percentages divide by all messages in each intent class, including rows with no token labels.
No T tag can also mean no token annotation; it does not establish that the message contains no toxic wording.
Use `intentClass` as the target for message classification. A rule that predicts toxicity whenever a T tag occurs will disagree with these labels.

## Data quality and context

{table(blank_table)}

- Duplicate Id values: **{int(df.Id.duplicated().sum()):,}**; exact duplicate rows: **{int(df.duplicated().sum()):,}**.
- Repeated utterance strings beyond their first occurrence: **{duplicate_messages:,}**. Repeated short chat messages are not automatically duplicate training records.
- No token labels: **{int(labels.map(len).eq(0).sum()):,}** rows. Inspect these before slot-model training; do not silently invent O labels.
- Unexpected slot classes beyond the six classes and SEPA: **{unexpected}**.
- Token count excluding SEPA: mean **{lengths.mean():.2f}**, median **{lengths.median():.0f}**, 95th percentile **{lengths.quantile(.95):.0f}**, maximum **{lengths.max()}**.
- Median messages per represented conversation: **{conversations.median():.0f}**. Training rows may provide only part of a conversation.

To inspect context, group by `(matchId, conversationId)` and sort by `chatTime`, then `Id` as a deterministic tie-breaker. Same-time order may remain ambiguous.
For a new context-aware split, keep each match together; before using the official splits, check their match overlap rather than assuming independence.
Never supply future messages as context for a live classifier.

## Connection to the microphone pipeline

This CSV is game chat, while `record_STT.py` produces spoken transcripts. This analysis does not train or attach a toxicity classifier.
Evaluate a future classifier on representative STT transcripts: punctuation, abbreviations and recognition errors can differ from the training text.
For binary message toxicity, E + I versus A + O is a possible mapping; preserve the original labels for analysis.

## Reproduce

Run `env/bin/python analyze_conda.py`. Optional: `--csv PATH --output DIRECTORY`.
Outputs include `overview.png`, `overview.pdf`, intent/token count CSVs, and the intent/token cross-table.
JSA evaluates the message class and every token label jointly; a message-only classifier needs separate evaluation. [Repository evaluation definition](https://github.com/usydnlp/CONDA#evaluation-metrics).
"""
    (output / "report.md").write_text(report)
    print(report)
    print(f"\nCharts and tables saved to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("CONDA_train.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "analysis")
    args = parser.parse_args()
    analyze(args.csv, args.output)
