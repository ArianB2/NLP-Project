# Beyond Profanity: Detecting Hostile Intent in Gaming Chat

Arian Berisha and Wadood Abdul, CSCI 49362 (NLP), Hunter College, Fall 2026

**Question:** Do toxicity classifiers rely on profanity more than on hostile intent in game chat, and does chat context help detect hostility aimed at other players?

**Task:** given a chat message, predict 1 if it's hostile toward a player, 0 if not. Swearing at the game, at Valve, at yourself, or friendly banter is 0.

**Data:** [CONDA](https://github.com/usydnlp/CONDA) (Dota 2 chat, Weld et al. 2021) plus about 300 messages we label by hand.

## Models

1. TF-IDF + logistic regression (message only)
2. unitary/toxic-bert, no training. Rule A: toxic ≥ 0.5. Rule B: insult or threat ≥ 0.5
3. DistilBERT fine-tuned on the message only
4. DistilBERT fine-tuned on the 3 earlier chat lines + the message

Models 1, 3 and 4 are trained on CONDA's labels (E and I = 1, A and O = 0). Our hand labels are only used for testing.

## Files

| File | What it does |
|---|---|
| `src/common.py` | shared paths, profanity check, metrics |
| `src/prepare_data.py` | cleans CONDA, makes labels, context and the train/dev/test split |
| `src/make_label_sheet.py` | makes the two labeling sheets |
| `src/agreement.py` | checks our agreement (Cohen's kappa) and makes `gold.csv` |
| `src/tfidf_baseline.py` | model 1 + top words chart |
| `src/toxicbert_baseline.py` | model 2 |
| `src/train_distilbert.py` | models 3 and 4 (`--context`) |
| `src/evaluate.py` | results tables, charts, confusion matrices, error examples |
| `tests/test_data.py` | sanity checks on the data |
| `app.py` | Gradio demo |
| `notebooks/run_pipeline_colab.ipynb` | runs everything on Colab |
| `LABELING_GUIDELINES.md` | our labeling rules |

The mic / speech-to-text files and the BART experiments are from earlier work and aren't part of this pipeline.

## How to run

On Colab: open `notebooks/run_pipeline_colab.ipynb`, set the runtime to T4 GPU, and run all cells.

Or locally:

```
pip install -r requirements-pipeline.txt
python src/prepare_data.py
python -m pytest tests/
python src/tfidf_baseline.py
python src/toxicbert_baseline.py
python src/train_distilbert.py
python src/train_distilbert.py --context
python src/evaluate.py
python app.py
```

Labeling: upload `data/labels/label_sheet_<name>.csv` to Google Sheets, label it, download it back as CSV with the same name, then run `python src/agreement.py`.

## Choices we made

- **Split by match.** CONDA's train and valid files share 1,772 matches and 3,540 conversations, so we combined them and split whole matches 80/10/10. The tests check that no match is in two splits.
- **Context** is the 3 previous lines in the same match, marked `self`, `ally` or `enemy`.
- **Profanity** means a word from `resources/profanity_patterns.txt`. Insults like "noob" aren't in the list on purpose.
- **Threshold** is 0.5 for every model, chosen before seeing results.

## Results so far

TF-IDF + LR on the test matches (CONDA labels, 3,803 messages):

| | Precision | Recall | F1 |
|---|---|---|---|
| All | 0.939 | 0.661 | 0.776 |
| With profanity (394) | 0.920 | 0.745 | 0.824 |
| No profanity (3,409) | 0.955 | 0.604 | 0.740 |

Recall is lower without profanity. Half of the top 30 hostile words are profanity, and some are nationalities ("peru", "russian"). See `results/figures/`.

## AI use

We used AI tools to explain concepts, summarize papers, and help with code. We reviewed all the code and can explain every part.
