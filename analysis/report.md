# Understanding the local CONDA training data

Generated from `CONDA_train.csv` by `analyze_conda.py`. These figures describe this file only.

## What a row represents

One row is a chat utterance. The paper merges consecutive messages from one player with `[SEPA]`.
It provides two tasks: human-annotated whole-message intent and lexicon-derived token labels.
The paper describes 44,869 utterances overall, split into 26,921 training and 8,974 each for validation/test.
Only the training CSV is present locally. [Paper, sections 3 and 5.1](https://arxiv.org/html/2106.06213v2).

## Local size and intent distribution

- Messages: **26,921**; columns: **10**.
- Matches represented: **1,911**; match/conversation pairs: **9,409**.
- E + I messages: **5,220 (19.39%)**.
- Always guessing O would score **74.22% accuracy** on this file. That illustrates why accuracy alone is misleading.

| Label | Meaning | Messages | Percent |
| --- | --- | --- | --- |
| O | Other | 19982 | 74.22 |
| E | Explicit toxicity | 3528 | 13.11 |
| I | Implicit toxicity | 1692 | 6.29 |
| A | Action | 1719 | 6.39 |

E describes overt toxicity; I describes contextual or hidden toxicity. A covers non-toxic actions such as reporting or pausing; O is the remaining category.
These are the dataset's annotation categories, not labels inferred by this script. [Annotation instructions](https://arxiv.org/html/2106.06213v2#S3.SS3).

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

| Label | Meaning | Tokens | Percent (excludes SEPA) |
| --- | --- | --- | --- |
| O | Other | 56823 | 63.7 |
| P | Pronoun | 12000 | 13.45 |
| S | Game slang | 10036 | 11.25 |
| C | Character/hero | 4781 | 5.36 |
| T | Toxicity | 4295 | 4.81 |
| D | Dota-specific | 1274 | 1.43 |

`C` means a game character/hero, not a character in a string. `O` means other, not numeric zero.
`SEPA` appears **10,419** times as a separator label and is excluded from the six semantic classes above.
The supplied tokens have separate preprocessing, so splitting raw `utterance` on spaces is not a reliable way to align them with `slotClasses`.
The paper describes lexicon matching for slots, rather than manual contextual judgment for every token. [Paper, section 3.3](https://arxiv.org/html/2106.06213v2#S3.SS3).

## Why message intent and toxic words differ

| Intent | No T token | Has T token | Percent with T token |
| --- | --- | --- | --- |
| O | 19233 | 749 | 3.75 |
| E | 762 | 2766 | 78.4 |
| I | 1631 | 61 | 3.61 |
| A | 1671 | 48 | 2.79 |

The percentages divide by all messages in each intent class, including rows with no token labels.
No T tag can also mean no token annotation; it does not establish that the message contains no toxic wording.
Use `intentClass` as the target for message classification. A rule that predicts toxicity whenever a T tag occurs will disagree with these labels.

## Data quality and context

| Column | Blank rows |
| --- | --- |
| Id | 0 |
| matchId | 0 |
| conversationId | 0 |
| utterance | 7 |
| chatTime | 0 |
| playerSlot | 0 |
| playerId | 8 |
| intentClass | 0 |
| slotClasses | 834 |
| slotTokens | 834 |

- Duplicate Id values: **0**; exact duplicate rows: **0**.
- Repeated utterance strings beyond their first occurrence: **7,978**. Repeated short chat messages are not automatically duplicate training records.
- No token labels: **834** rows. Inspect these before slot-model training; do not silently invent O labels.
- Unexpected slot classes beyond the six classes and SEPA: **[]**.
- Token count excluding SEPA: mean **3.31**, median **2**, 95th percentile **10**, maximum **51**.
- Median messages per represented conversation: **1**. Training rows may provide only part of a conversation.

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
