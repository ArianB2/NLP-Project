# HIC project progress

Updated: November 18, 2026.

**Active project: `/Volumes/Samsung T7/HIC`.** Earlier work was accidentally written to `/Users/wadoodabdul/dev/HIC` and transferred to the active project on November 12. Future changes belong in the external-drive project.

This document consolidates the recorded implementation, data analysis, annotation work, BART experiments, findings and next steps. It is based on the project files, `session.md`, and the experiment tables supplied in chat. Historical checks are reported as historical results; no API calls, microphone recordings, BART inference or training were run while writing this document. The full supplied experiment tables are preserved in Appendix A.

## 1. Project objective and current position

The objective is to classify hostility communicated toward another person or group in gaming conversation, including insults, humiliation, harassment, identity attacks, threats and encouragement of harm. The project should recognize implicit hostility without depending on swear words, and avoid treating ordinary gameplay, self-directed frustration, technical problems or friendly language as interpersonal attacks.

The intended pipeline is:

`Mac microphone → OpenAI realtime transcription → final utterance plus earlier context → hostile-intent classifier`

The microphone/transcription components are implemented. The classification work currently consists of BART zero-shot experiments, a shared annotation contract, a 240-message annotation pilot, and a fine-tuning plan. **A trained project-specific hostile-intent model has not been produced or connected to live transcription.**

## 2. Recorded timeline

| Date | Work recorded | Outcome |
| --- | --- | --- |
| November 9, 2026 | Project inspection, requirements, OpenAI documentation and Mac STT implementation | Microphone/service/pipeline modules created; five offline tests passed |
| November 9, 2026 | Initial OpenAI connection check | Reached the API, which returned no credits remaining |
| November 9, 2026 | Connection retry after credits were added | Authenticated transcription session accepted and closed cleanly; no audio sent in this check |
| November 9, 2026 | Simple launch command and terminal transcription | `./start` added; partial/final transcript display implemented |
| November 9, 2026 | CONDA exploration and annotation-contract draft | Report, charts, summary CSVs, schema and synthetic examples created |
| November 11, 2026 | Stratified CONDA annotation pilot | 240 real messages sampled and annotated as model-assisted proposals |
| November 11, 2026 | Historical BART snapshot and training plan | Five-hypothesis dev-directory baseline preserved; no weight changes |
| November 12, 2026 | Correct working directory restored | Missing artifacts transferred to external drive; its six-label BART source preserved |
| Reported in the November 12 discussion; actual run dates not recorded | Three BART-large-MNLI experiments supplied by the user | 11 messages per experiment, with top-three predictions and scores; reviewed without rerunning |
| November 18, 2026 | Progress consolidation | This document created; historical evidence and pending work distinguished |

## 3. Mac microphone and OpenAI STT

### Implementation

| File | Responsibility |
| --- | --- |
| `openai_service.py` | Loads `.env`, authenticates over WebSocket, configures transcription, sends PCM audio, commits turns, processes errors, and retains transcripts by item ID |
| `mic.py` | Starts/stops native macOS microphone capture through sounddevice/PortAudio Core Audio; passes frames safely into asyncio |
| `record_STT.py` | Connects microphone capture and transcription, detects speech/silence, and prints partial/final text to the terminal |
| `main.py` | Entry point that launches the recording/STT pipeline |
| `start` | Uses the project environment to run `main.py` without manual environment activation |
| `test_stt.py` | Offline protocol and pipeline checks using simulated audio/API events |
| `requirements.txt` | Dependency inventory; initial audio dependencies were verified and data-analysis libraries added |

Current service settings in the inspected code:

- Model: `gpt-live-transcribe`; connection: `wss://api.openai.com/v1/realtime?intent=transcription`.
- Input: mono, signed 16-bit little-endian PCM, 24 kHz, 20 ms frames; audio travels as base64 JSON messages.
- Low transcript delay by default; optional language hints and alternative delay levels are exposed by the CLI.
- Simple local energy threshold of 500, 200 ms pre-roll, 500 ms silence before committing a turn, and a 30-second turn cap.
- Ctrl+C stops capture, sends buffered speech and waits up to 15 seconds for final transcription of committed turns.
- Short nonempty turns are padded to 100 ms before commitment. Final transcripts are assembled in commit order even when completions arrive out of order.
- Transcripts are printed and retained in memory; the implementation does not save audio or transcript files.
- Microphone implementation currently supports macOS; Windows support remains pending.

### Recorded checks and limits

Five offline tests passed: session configuration/cleanup, audio encoding/short-turn padding and out-of-order final text, API error propagation, pre-roll/silence/stop flushing, and missing-key validation. CLI help and dependency consistency were also checked during implementation.

The first connection-only API test failed because the account had no credits. After credits were added, the retry succeeded. This verifies session authentication/configuration at that time, not current account balance, recognition accuracy, or end-to-end microphone latency. No recorded live microphone benchmark establishes the actual STT latency yet.

## 4. CONDA dataset investigation

Reference: [CONDA paper](https://arxiv.org/html/2106.06213v2) and [source repository](https://github.com/usydnlp/CONDA). The following local findings are recorded in `analysis/report.md`.

The full dataset described in the paper contains 44,869 utterances. The local `CONDA_train.csv` contains the 26,921-message training split, not the entire train/validation/test collection. Local validation and test CSVs have not been added in the recorded work.

| Source intent | Meaning | Local messages | Share |
| --- | --- | ---: | ---: |
| O | Other | 19,982 | 74.22% |
| E | Explicit toxicity | 3,528 | 13.11% |
| I | Implicit toxicity | 1,692 | 6.29% |
| A | Action | 1,719 | 6.39% |

E + I totals 5,220 messages, or 19.39%. The file represents 1,911 matches and 9,409 match/conversation pairs, with 10 columns.

The two annotation levels must remain distinct:

- `intentClass` describes the whole utterance. Source O means the message does not meet the source E, I or A criteria; it may still contain profanity.
- `slotClasses` labels processed tokens. Slot O means a token did not match the source lexicons, not that it is guaranteed harmless.
- Slots C, D, S, P, T and O mean character/hero, Dota-specific, game slang, pronoun, toxicity and other. Their annotations are not equivalent to our hostility ground truth.
- `[SEPA]` separates consecutive messages merged by the source. It has a separator label and is excluded from the six semantic token classes in our summaries.

| Semantic slot | Local tokens | Share excluding SEPA |
| --- | ---: | ---: |
| O: other | 56,823 | 63.70% |
| P: pronoun | 12,000 | 13.45% |
| S: game slang | 10,036 | 11.25% |
| C: character/hero | 4,781 | 5.36% |
| T: toxicity | 4,295 | 4.81% |
| D: Dota-specific | 1,274 | 1.43% |

There are another 10,419 SEPA separator labels. Message counts and token counts are different units and should not be compared as if each token were a separate message.

### Quality findings

- 834 rows have no token annotations; seven utterances are blank; eight player identifiers are blank.
- No duplicate utterance IDs or exact duplicate rows were found.
- 7,978 utterance strings repeat beyond their first occurrence. Common short messages are not automatically invalid duplicates.
- Mean processed token count excluding SEPA is 3.31, median 2, 95th percentile 10, maximum 51.
- The shuffled training split does not guarantee complete conversation history.
- In the local file, 78.40% of source E messages have a T token, versus 3.61% of source I and 3.75% of source O. Missing token annotations complicate absence claims.

### Interpretation and artifacts

The charts demonstrate imbalance; they do not by themselves establish whether there are enough examples for a particular model. Always guessing O would yield 74.22% accuracy on this training file, so class-level recall/F1 and false-positive analysis matter. Slang such as gg or wp is not automatically hostile.

Installed pandas and matplotlib, which supplies `matplotlib.pyplot`; NumPy was already installed. Added `analyze_conda.py` and generated:

- `analysis/report.md` — definitions, counts, quality findings and implications.
- `analysis/overview.png` and `analysis/overview.pdf` — class distribution, token labels, lengths and intent/token relationship.
- `analysis/intent_counts.csv`, `slot_counts.csv`, and `intent_vs_toxic_tokens.csv` — reusable summaries.

`clean_dataset.py` is still empty. No bulk cleaning, Dota-term removal or multi-dataset merge has been implemented.

## 5. Common data and annotation contract

Created `DATA_CONTRACT.md`, `data_contract.schema.json` and three invented illustrations in `data_contract.examples.jsonl`. The examples are explicitly excluded from training. Contract version is 0.1.0 and remains a draft.

| Dimension | Contract meaning |
| --- | --- |
| hostility | hostile, non_hostile, uncertain; null means not yet annotated |
| expression | explicit, implicit, not_applicable, unknown |
| target | Person/group attacked, distinct from who simply receives a message |
| behaviors | Insult, humiliation, harassment, identity attack, threat, encouragement of harm |
| profanity | Present, absent or unknown, assessed independently of hostility |
| context | Available earlier messages only, with history completeness recorded |
| evidence/rationale | Optional support spans and annotation explanation; not model inputs |
| provenance | Source IDs/labels/version/split, game, modality, raw text and transformations |
| review/quality | Annotation method and status, review identifiers, inclusion decision and reasons |

Decisions established in the draft:

- Preserve original text and source labels; do not silently replace source E/I with our reviewed labels.
- Unknown and uncertain are different from non-hostile. Do not turn ambiguous sarcasm or missing history into neutral training examples.
- Profanity, negative sentiment, game combat and criticism are insufficient on their own to establish interpersonal hostility.
- Keep game-specific words in raw text. Make labels game-independent; evaluate actual cross-game generalization rather than assuming that deleting Dota columns achieves it.
- Keep filtering reversible and retain reasons. Group new splits by match/session and audit duplicates, edited variants and context dependencies.
- Only reviewed/adjudicated records with an include quality decision qualify for the first supervised export. Uncertain records remain outside the initial binary loss unless resolved.

## 6. CONDA annotation pilot

Pilot directory: `data/annotations/conda_pilot_v1/`.

Sampled 240 distinct nonblank utterances with seed 20261111: 80 source E, 80 source I, 60 source O and 20 source A. Case/whitespace-normalized repeats were excluded from candidate pools for diversity. Source IDs, selection rules and source SHA-256 were saved. This is deliberately stratified, not a sample of natural deployment prevalence.

Each selected message received an individually authored model-assisted proposal and rationale against the contract. Available context comprises up to five same-match/same-conversation messages with strictly earlier chat times. Tied-time messages were excluded because their order is not established. There are 145 target records with earlier context and 219 distinct matches.

| Proposed category | Messages |
| --- | ---: |
| Explicit hostility | 61 |
| Implicit hostility | 34 |
| Non-hostile | 79 |
| Uncertain | 66 |
| Total | 240 |

These correspond to 95 hostile, 79 non-hostile and 66 uncertain primary labels. Profanity annotations are 41 present, 188 absent and 11 unknown. Some source E/I labels were not retained as project-hostile because the current text/context did not establish an attack. Many short ez variants remain uncertain rather than being automatically called implicit hostility.

All 240 annotations have method `model_assisted`, status `proposed`, and quality `pending`. No BART model generated these labels. They are not independently human-reviewed gold. There are 174 non-uncertain candidates for review and zero currently approved training records under the contract. Annotation uncertainty, multilingual wording and threat/gameplay boundaries still require review.

| Artifact | Purpose |
| --- | --- |
| `sentences_labels.csv` | Sentence and primary label, plus expression/category, context and rationale |
| `annotations.jsonl` | Complete contract records preserving original source annotations |
| `decisions.tsv` | Authored per-message annotation decisions |
| `selected_source_rows.json` | Selected source rows and available earlier rows |
| `sample_manifest.json` | Seed, sampling policy, IDs and source hash |
| `summary.json` | Counts and recorded validation results |
| `README.md` | Sampling/annotation policy and limitations |
| `build_pilot.py` | Rebuilds exports and validates source/schema/annotation consistency |

Recorded checks covered 240 distinct source IDs, exact source-text equality, reproducible selected IDs, CSV/JSONL agreement, structural contract validation, semantic label consistency, evidence bounds and earlier-only context. These checks verify structure and traceability, not annotation correctness or agreement between people.

## 7. Three BART-large-MNLI hypothesis experiments

Model reported: `facebook/bart-large-mnli`. These are zero-shot inference/hypothesis experiments, not documented fine-tuning or weight-update runs. Results were supplied by the user and were not rerun to produce this report. Each experiment contains 11 messages and three displayed predictions per message: 33 reported prediction rows per experiment, 99 across the three experiments.

### Experiment 1 — full-sentence criticism hypotheses

The reported labels are consistent with the historical five-hypothesis source snapshot:

1. The speaker is criticizing themself
2. The speaker is criticizing the video game
3. The speaker is criticizing another player
4. The speaker is criticizing the team
5. The speaker is not criticizing anyone

The matching saved source uses template `{}.` and `multi_label=False`. The pasted outputs alone do not independently prove the complete run configuration or model revision.

Observed top predictions include self for both first-person performance complaints, player for the two direct player criticisms, team for the team complaint, and no criticism for the explicit denial. Failure cases include stupid tree as player, lagging as self, and praise as criticism of the game.

### Experiment 2 — broader game-related and connection alternatives

The supplied predictions introduce The speaker is criticizing something in the game and The speaker is criticizing the internet connection. The complete candidate list/order, exact hypothesis template, settings and immutable model revision were not supplied, and have not been reconstructed as facts.

The broad game-related label becomes top-ranked for self-performance complaints and stupid tree. It also absorbs the praise example. The technical-problem example still ranks self first; the connection hypothesis appears only third at 10.64%.

Interpretation: the broad game-related category overlaps with several specific criticism targets. This is a hypothesis-design concern, not measured evidence of a learned improvement.

### Experiment 3 — explicit criticism-target template

The current external-drive `Bart-mnli/bart.py` is consistent with this experiment and uses these labels:

1. the speaker themselves
2. another player
3. the entire team
4. the video game or its environment
5. a technical problem
6. nobody in particular

Template: `The criticism is directed at {}.`; `multi_label=False`. The file shows the current configuration; the supplied historical results have not been independently replayed at a pinned revision.

The two direct player criticisms receive player scores of 98.19% and 96.18%. Stupid tree changes to game/environment at 60.59%. However, lagging and the missed-shot complaint rank another player first, and the praise example still ranks game/environment first. The first-person bad-play statement ranks game/environment at 30.76%, just above self at 30.05%.

### Cross-experiment findings

- Wording changes help some examples and hurt others. The three tables do not establish an overall winning configuration.
- Source labels describe criticism targets, while our main target is hostility. Constructive criticism can target a teammate without being hostile; sarcastic praise may communicate hostility.
- The criticism-directed template starts from the assumption that criticism is present, which is a plausible contributor to misreading praise; this is a hypothesis to test, not a proven causal result.
- The lagging example illustrates difficulty separating profanity/technical frustration from an attack on another player.
- The supplied scores are relative label scores under the experiment configuration, not calibrated probabilities of hostile intent. Scores from different candidate sets should not be interpreted as directly comparable confidence gains.
- Eleven diagnostic messages without an independently annotated evaluation set do not establish accuracy, F1, implicit-hostility recall or cross-game generalization.
- The text of I'm such an idiot has a terminal period in Experiment 3 but not in the earlier outputs. Future controlled comparisons should hold all input text and punctuation fixed.
- Only top-three outputs are available, not complete score vectors, raw logits, run timings, seed/device records or a pinned model revision for all three runs.

Full tables and original score values are preserved in Appendix A.

## 8. Baseline preservation and directory correction

On November 11, the five-hypothesis source in the dev-directory copy was frozen under `Bart-mnli/baselines/2026-11-11/`:

- `bart.py.snapshot` — exact historical source.
- `hypothesis.json` — candidate labels, order, template, example messages and SHA-256.
- `environment_packages.json` — package inventory at capture time.

The original source did not pin a model/tokenizer revision, so this freezes the source and hypothesis configuration, not a fully reproducible weight checkpoint. No training or weight modification occurred during snapshotting.

On November 12, the working-directory mistake was corrected. Earlier STT, contract and analysis files already matched on the external drive. The missing annotation pilot, historical baseline, training plan and session history were transferred and verified. The external-drive `bart.py`, containing different six-label hypotheses, was preserved. The training plan now marks its five-label baseline as historical. The transferred builder verifies the historical snapshot hash and reports live BART differences without rejecting annotation validation. `.env`, environment files, Git metadata and the source CSV were not overwritten. Transfer details are in `transfer_2026-11-12.json`.

## 9. Fine-tuning and weight-modification plan — not executed

The detailed plan is in `Bart-mnli/TRAINING_PLAN.md`.

Recommended sequence:

1. Review/adjudicate the 240 proposed annotations and refine the guidelines.
2. Expand reviewed data, with roughly 2,000 messages suggested as an initial collection target, not a guarantee of adequate performance.
3. Include other-game and final STT examples, non-hostile profanity, profanity-free hostility, constructive criticism, gameplay violence, praise and sarcasm.
4. Preserve BART zero-shot baselines. Pin model/tokenizer revisions and record complete predictions and run configuration before evaluation.
5. Fine-tune a compact pretrained encoder, initially DistilBERT, with a new three-class head: non_hostile, explicit_hostility and implicit_hostility. Compare a binary hostility alternative if appropriate.
6. Train the new head with the encoder frozen, then unfreeze the top two blocks; consider full fine-tuning only if validation supports it.
7. Compare a separate BART direct classifier or LoRA adaptation if quality gains justify its cost. A newly initialized project head is required for the direct-classifier route; the three MNLI outputs are not our three project labels simply because the dimensions match.
8. If retaining NLI instead, use separately reviewed premise/hypothesis relationship labels. Uncertain hostility is not automatically the NLI neutral class.

Proposed compact-classifier search settings: 256-token context/current-message budget, head learning rate 1e-4 to 5e-4, unfrozen blocks 1e-5 to 3e-5, effective batch 16, weight decay 0.01, up to 3–5 epochs with early stopping, and multiple fixed seeds. Optional BART LoRA candidates are rank 8/16, alpha 16/32 and dropout 0.05 with actual module names verified before implementation. These settings have not been tested.

Evaluation should include match/session-disjoint splits, duplicate and context-dependency audits, per-class precision/recall/F1, macro-F1, confusion matrices, profanity/context slices, held-out games, calibration/abstention and end-to-end timing on intended hardware. Natural prevalence matters for deployment precision; the deliberately stratified pilot cannot estimate it.

No training entry point, fitted model, learned adapter, weight-update result or held-out performance report has been produced in the recorded work. Multiplying output scores would be a decision-rule change, not model fine-tuning.

## 10. Remaining work and next checkpoints

- Human-review the pilot, especially the 66 uncertain cases, implicit cases, multilingual text and threat/gameplay boundaries; measure agreement and adjudicate disputes.
- Settle the contract definitions and annotation guideline version before large-scale relabeling.
- Create a production-quality ingestion/validation path beyond the pilot builder; the existing schema checks cover its current keyword subset, not a universal validator.
- Research and assess additional datasets, licenses, context availability and label compatibility. No additional datasets have been imported in the recorded work.
- Implement reversible filtering after review. Preserve raw game terms; test any entity masking as a separate experiment rather than deleting them.
- Obtain/audit evaluation splits, group and deduplicate records including context dependencies, and reserve an untouched test set.
- Run controlled criticism-target and hostility experiments separately, with fixed messages and exact hypothesis configurations.
- Implement the selected fine-tuning experiment only as a new training component; retain original BART baselines.
- Run a live microphone transcription/latency evaluation, then evaluate the classifier on representative STT text before integration.
- Add Windows microphone capture when requested.

## 11. Useful commands and file references

Run from the active project directory:

```sh
cd "/Volumes/Samsung T7/HIC"
./start
./start --list-devices
env/bin/python -m unittest -v test_stt.py
env/bin/python analyze_conda.py
env/bin/python data/annotations/conda_pilot_v1/build_pilot.py
```

`./start` opens the microphone/transcription pipeline; it may need microphone permission and an API key with available credit. It does not run the BART classifier. Ctrl+C stops recording and prints the full transcript.

Environment note: this exFAT drive can add macOS `._*` metadata files to `env`, which crashes `matplotlib` and `transformers` on import. After any `pip install`, run `find env -name '._*' -type f -delete` from the project directory.

`env/bin/python Bart-mnli/bart.py` runs the current six-label demonstration separately and may download model files if uncached. It is an inference demonstration, not a training command. None of these commands was executed while preparing this document.

Relevant project references:

- [Session history](session.md)
- [Data analysis](analysis/report.md)
- [Analysis charts](analysis/overview.png)
- [Data contract](DATA_CONTRACT.md)
- [Contract schema](data_contract.schema.json)
- [Annotation pilot policy](data/annotations/conda_pilot_v1/README.md)
- [Sentence-and-label CSV](data/annotations/conda_pilot_v1/sentences_labels.csv)
- [Full annotations](data/annotations/conda_pilot_v1/annotations.jsonl)
- [Training plan](Bart-mnli/TRAINING_PLAN.md)
- [Historical hypotheses](Bart-mnli/baselines/2026-11-11/hypothesis.json)

## Appendix A — complete user-supplied BART experiment results

The following tables preserve the supplied labels, rankings, percentages and comparison notes. Only heading depth has been adjusted for this document. They are reported experiment outputs, not newly generated or independently verified predictions. The experiment dates/configuration gaps discussed above remain unresolved.

### Experiment 1

| Message                                 | Rank | Predicted label                           | Score  |
| --------------------------------------- | ---- | ----------------------------------------- | ------ |
| I can't believe how bad I played today. | 1    | The speaker is criticizing themself       | 43.87% |
|                                         | 2    | The speaker is criticizing another player | 26.89% |
|                                         | 3    | The speaker is criticizing the team       | 20.18% |
| This game is terrible.                  | 1    | The speaker is criticizing the video game | 64.53% |
|                                         | 2    | The speaker is criticizing the team       | 24.43% |
|                                         | 3    | The speaker is criticizing another player | 6.96%  |
| That player is so annoying.             | 1    | The speaker is criticizing another player | 82.37% |
|                                         | 2    | The speaker is criticizing the team       | 11.28% |
|                                         | 3    | The speaker is criticizing the video game | 3.31%  |
| Our team is a mess.                     | 1    | The speaker is criticizing the team       | 77.16% |
|                                         | 2    | The speaker is criticizing another player | 12.76% |
|                                         | 3    | The speaker is criticizing themself       | 8.80%  |
| I'm not mad at anyone.                  | 1    | The speaker is not criticizing anyone     | 99.71% |
|                                         | 2    | The speaker is criticizing themself       | 0.12%  |
|                                         | 3    | The speaker is criticizing another player | 0.07%  |
| I'm such an idiot                       | 1    | The speaker is criticizing themself       | 80.68% |
|                                         | 2    | The speaker is criticizing another player | 10.28% |
|                                         | 3    | The speaker is criticizing the team       | 5.10%  |
| stupid tree                             | 1    | The speaker is criticizing another player | 40.95% |
|                                         | 2    | The speaker is criticizing themself       | 36.39% |
|                                         | 3    | The speaker is criticizing the team       | 17.47% |
| my shit is lagging                      | 1    | The speaker is criticizing themself       | 62.45% |
|                                         | 2    | The speaker is criticizing another player | 27.80% |
|                                         | 3    | The speaker is criticizing the team       | 6.62%  |
| Great game guys                         | 1    | The speaker is criticizing the video game | 24.33% |
|                                         | 2    | The speaker is criticizing themself       | 23.25% |
|                                         | 3    | The speaker is not criticizing anyone     | 22.11% |
| Player 2 is absolutely dog shit         | 1    | The speaker is criticizing another player | 64.97% |
|                                         | 2    | The speaker is criticizing the video game | 20.66% |
|                                         | 3    | The speaker is criticizing the team       | 7.61%  |
| Can't believe I missed that shot        | 1    | The speaker is criticizing themself       | 43.44% |
|                                         | 2    | The speaker is criticizing another player | 35.18% |
|                                         | 3    | The speaker is criticizing the team       | 17.16% |

### Experiment 2

| Message                                 | Rank | Predicted label                                    | Score  |
| --------------------------------------- | ---- | -------------------------------------------------- | ------ |
| I can't believe how bad I played today. | 1    | The speaker is criticizing something in the game   | 39.18% |
|                                         | 2    | The speaker is criticizing themself                | 26.43% |
|                                         | 3    | The speaker is criticizing another player          | 16.20% |
| This game is terrible.                  | 1    | The speaker is criticizing something in the game   | 54.02% |
|                                         | 2    | The speaker is criticizing the video game          | 29.58% |
|                                         | 3    | The speaker is criticizing the team                | 11.20% |
| That player is so annoying.             | 1    | The speaker is criticizing another player          | 76.45% |
|                                         | 2    | The speaker is criticizing the team                | 10.47% |
|                                         | 3    | The speaker is criticizing something in the game   | 6.93%  |
| Our team is a mess.                     | 1    | The speaker is criticizing the team                | 48.37% |
|                                         | 2    | The speaker is criticizing something in the game   | 36.97% |
|                                         | 3    | The speaker is criticizing another player          | 8.00%  |
| I'm not mad at anyone.                  | 1    | The speaker is not criticizing anyone              | 99.47% |
|                                         | 2    | The speaker is criticizing something in the game   | 0.19%  |
|                                         | 3    | The speaker is criticizing themself                | 0.12%  |
| I'm such an idiot                       | 1    | The speaker is criticizing themself                | 73.64% |
|                                         | 2    | The speaker is criticizing another player          | 9.38%  |
|                                         | 3    | The speaker is criticizing something in the game   | 5.47%  |
| stupid tree                             | 1    | The speaker is criticizing something in the game   | 33.98% |
|                                         | 2    | The speaker is criticizing another player          | 26.49% |
|                                         | 3    | The speaker is criticizing themself                | 23.54% |
| my shit is lagging                      | 1    | The speaker is criticizing themself                | 50.56% |
|                                         | 2    | The speaker is criticizing another player          | 22.51% |
|                                         | 3    | The speaker is criticizing the internet connection | 10.64% |
| Great game guys                         | 1    | The speaker is criticizing something in the game   | 21.33% |
|                                         | 2    | The speaker is criticizing the video game          | 17.89% |
|                                         | 3    | The speaker is criticizing themself                | 17.09% |
| Player 2 is absolutely dog shit         | 1    | The speaker is criticizing another player          | 55.71% |
|                                         | 2    | The speaker is criticizing the video game          | 17.71% |
|                                         | 3    | The speaker is criticizing something in the game   | 13.85% |
| Can't believe I missed that shot        | 1    | The speaker is criticizing something in the game   | 33.18% |
|                                         | 2    | The speaker is criticizing themself                | 28.45% |
|                                         | 3    | The speaker is criticizing another player          | 23.05% |

### Experiment 3

| Message                                 | Rank | Predicted label                   | Score  |
| --------------------------------------- | ---- | --------------------------------- | ------ |
| I can't believe how bad I played today. | 1    | the video game or its environment | 30.76% |
|                                         | 2    | the speaker themselves            | 30.05% |
|                                         | 3    | another player                    | 21.13% |
| This game is terrible.                  | 1    | the video game or its environment | 70.04% |
|                                         | 2    | another player                    | 14.54% |
|                                         | 3    | the entire team                   | 4.72%  |
| That player is so annoying.             | 1    | another player                    | 98.19% |
|                                         | 2    | the speaker themselves            | 0.81%  |
|                                         | 3    | the video game or its environment | 0.58%  |
| Our team is a mess.                     | 1    | the entire team                   | 38.95% |
|                                         | 2    | another player                    | 19.15% |
|                                         | 3    | the speaker themselves            | 13.44% |
| I'm not mad at anyone.                  | 1    | nobody in particular              | 76.06% |
|                                         | 2    | a technical problem               | 6.77%  |
|                                         | 3    | the speaker themselves            | 6.07%  |
| I'm such an idiot.                      | 1    | the speaker themselves            | 39.33% |
|                                         | 2    | another player                    | 26.35% |
|                                         | 3    | nobody in particular              | 13.04% |
| stupid tree                             | 1    | the video game or its environment | 60.59% |
|                                         | 2    | another player                    | 18.15% |
|                                         | 3    | a technical problem               | 12.54% |
| my shit is lagging                      | 1    | another player                    | 33.22% |
|                                         | 2    | a technical problem               | 26.71% |
|                                         | 3    | the speaker themselves            | 25.85% |
| Great game guys                         | 1    | the video game or its environment | 38.63% |
|                                         | 2    | another player                    | 22.76% |
|                                         | 3    | the speaker themselves            | 16.27% |
| Player 2 is absolutely dog shit         | 1    | another player                    | 96.18% |
|                                         | 2    | the video game or its environment | 1.83%  |
|                                         | 3    | the speaker themselves            | 1.45%  |
| Can't believe I missed that shot        | 1    | another player                    | 30.79% |
|                                         | 2    | the speaker themselves            | 22.29% |
|                                         | 3    | the video game or its environment | 19.23% |

### Comparison: Top prediction across all three experiments

This compact table is useful for comparing how changes to the candidate labels affected the model.

| Message                                | Experiment 1    | Experiment 2          | Experiment 3          |
| -------------------------------------- | --------------- | --------------------- | --------------------- |
| I can't believe how bad I played today | Self (43.87%)   | Game-related (39.18%) | Game-related (30.76%) |
| This game is terrible                  | Game (64.53%)   | Game-related (54.02%) | Game-related (70.04%) |
| That player is so annoying             | Player (82.37%) | Player (76.45%)       | Player (98.19%)       |
| Our team is a mess                     | Team (77.16%)   | Team (48.37%)         | Team (38.95%)         |
| I'm not mad at anyone                  | None (99.71%)   | None (99.47%)         | None (76.06%)         |
| I'm such an idiot                      | Self (80.68%)   | Self (73.64%)         | Self (39.33%)         |
| stupid tree                            | Player (40.95%) | Game-related (33.98%) | Game-related (60.59%) |
| my shit is lagging                     | Self (62.45%)   | Self (50.56%)         | Player (33.22%)       |
| Great game guys                        | Game (24.33%)   | Game-related (21.33%) | Game-related (38.63%) |
| Player 2 is absolutely dog shit        | Player (64.97%) | Player (55.71%)       | Player (96.18%)       |
| Can't believe I missed that shot       | Self (43.44%)   | Game-related (33.18%) | Player (30.79%)       |

Note: The comparison table uses shortened, normalized category names for readability. The three detailed tables preserve the original prediction labels.

The exact complete candidate-label lists for Experiments 1 and 2 were not included in the outputs you pasted, so I haven't reconstructed them.
