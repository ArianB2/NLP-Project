# Hostile intent model plan — November 11, 2026

> Transfer note — November 12, 2026: this plan and its November 11 snapshot were authored against `/Users/wadoodabdul/dev/HIC`. The active project is `/Volumes/Samsung T7/HIC`, whose current `bart.py` uses a different six-label hypothesis configuration. That file has been preserved. The five-label baseline below is historical; recommendations remain a plan, not changes to the current BART code.

Planning only. `bart.py` remains unchanged. No inference run, model download, weight update or training experiment was performed for this task.

## 1. Freeze the current hypothesis baseline

The exact source is saved in `baselines/2026-11-11/bart.py.snapshot`; its SHA-256 and configuration are in `hypothesis.json` alongside the original 11 demonstration messages.

Current model: `facebook/bart-large-mnli`; pipeline: zero-shot classification; `multi_label=False`; hypothesis template: `{}.`. Candidate hypotheses, in their original order:

1. The speaker is criticizing themself
2. The speaker is criticizing the video game
3. The speaker is criticizing another player
4. The speaker is criticizing the team
5. The speaker is not criticizing anyone

Do not edit this list to improve pilot results and still call it the original baseline. Any future hypothesis changes get a new experiment ID. Freezing hypotheses here means preserving this experimental configuration, not setting gradient flags or claiming a frozen weight checkpoint.

The source does not pin a Hugging Face revision. The snapshot therefore freezes source/hypotheses, but does not by itself guarantee identical future weights/tokenization. Before the first scored baseline run, record an immutable model/tokenizer commit, installed package versions, device, precision and actual run outputs. An environment package inventory is saved beside the snapshot; no claim is made about what revision a previous local run used.

## 2. Diagnose the task mismatch

The current labels describe the object of criticism. Our contract describes hostile intent. Constructive criticism of a teammate can be non-hostile, while mock praise can communicate hostility. Criticism of a game or self is also distinct from an attack on another person. The current script has no previous-message input, uncertainty handling, or contract label export.

BART-MNLI evaluates premise/hypothesis entailment. With the current single-label setup it ranks the supplied alternatives; a high relative score is not a calibrated probability of hostile intent. [Model card](https://huggingface.co/facebook/bart-large-mnli), [pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline).

Do not manually multiply the team/player scores and describe this as modifying learned weights. Such post-processing changes decision rules, not model representations. Nor can we evaluate the five criticism labels directly against hostility labels: a separate criticism annotation would be needed. A coarse player/team-to-hostile mapping, if explored, must be explicitly labeled an exploratory heuristic and audited for constructive-criticism errors.

## 3. Data preparation before training

Use the 240-record pilot in `../data/annotations/conda_pilot_v1/` to refine the guidelines and review disagreement. It currently contains 61 explicit-hostile, 34 implicit-hostile, 79 non-hostile and 66 uncertain proposed examples. None is human-reviewed. This is enough to inspect failure modes, not evidence that a model can generalize to unseen games.

Independently double-annotate the pilot, especially implicit cases, report requests, identity references, multilingual text, and non-hostile profanity. Adjudicate disputes and version any guideline changes. Do not convert uncertain labels to neutral or use source E/I as a replacement for review.

Then expand reviewed data, initially targeting roughly 2,000 diverse messages if annotation capacity permits. This is a collection target, not a guaranteed minimum for acceptable performance. Include non-hostile profanity, profanity-free insults, self/game frustration, tactical killing language, constructive criticism, genuine praise and sarcastic praise. Collect other-game and representative final STT transcripts separately; preserve original text and source labels. Keep uncertain cases for abstention analysis, outside the initial supervised loss.

Audit official validation/test data before use. Preserve official split provenance; inspect match, conversation, exact/near-duplicate and context overlap. Do not assume a source's official split is match-disjoint. If overlap prevents an independent context-aware test, report the limitation and build a separately named match-disjoint evaluation set. For new data, assign match/session and duplicate/edited-example groups before splitting. Keep all target messages and their context dependencies from crossing partitions. If the small pilot is used during model/hypothesis design, it is development data, never an untouched final test.

## 4. Recommended first trainable model

Fine-tune a pretrained compact encoder, initially `distilbert/distilbert-base-uncased`, with a new three-class sequence-classification head:

| Class | Contract export |
| --- | --- |
| non_hostile | hostility non_hostile; expression not_applicable |
| explicit_hostility | hostility hostile; expression explicit |
| implicit_hostility | hostility hostile; expression implicit |

This creates our own project checkpoint by fine-tuning; it does not train a language model from scratch. The model card and official sequence-classification recipe support adapting this pretrained encoder with task labels. [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased), [fine-tuning recipe](https://huggingface.co/docs/transformers/tasks/sequence_classification).

Input: clearly delimited earlier context followed by the current message; pseudonymous speaker markers distinguish turns. Start with a 256-token budget, always retaining the current message and truncating oldest context first. Do not pass source labels, rationales, evidence, profanity annotations, source intent, reviewer metadata or dataset IDs as input. Compare current-message-only versus context input with the same split. Context-dependent examples cannot be treated as reliable sentence-only gold without reporting missing information.

Weight-update stages:

1. Freeze the pretrained encoder and train only the newly initialized classification head as a low-capacity reference.
2. Unfreeze the top two transformer blocks plus the head; keep the remainder frozen.
3. Consider full encoder fine-tuning only when reviewed data and grouped validation show an improvement over the simpler stages.

Initial search proposal, not a promised optimum: head learning rate 1e-4 to 5e-4, unfrozen encoder blocks 1e-5 to 3e-5, effective batch 16, weight decay 0.01, and at most 3–5 epochs with early stopping on validation macro-F1. Use training-only class counts for weighted cross-entropy; compare against unweighted loss rather than combining aggressive oversampling and weights blindly. Run several fixed seeds, initially 13/42/2026. Select settings on validation only and retain learning curves to identify overfitting.

## 5. BART adaptation alternative

Keep the frozen original baseline and create a separate experiment directory if BART is adapted. Two different routes are possible:

**Direct classifier:** start from the BART checkpoint with a newly initialized three-class project head. Its MNLI head also has three outputs, so matching dimensions are not a reason to reuse its semantics. Explicitly reinitialize the task head; do not just rename entailment/neutral/contradiction to our labels. Train the head first, then compare LoRA with selective/full fine-tuning. For LoRA, freeze base weights and add trainable low-rank updates to the selected attention projection matrices; train/save the project classification head as well. Initial adapter candidates: rank 8 or 16, alpha 16 or 32, dropout 0.05, query/value projections. Verify actual module names and trainable parameter counts before implementing; include encoder/decoder selection explicitly in the run config. [LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora).

**Retain NLI:** create a new, separately versioned set of contract-aligned hypotheses and annotate premise/hypothesis relationships. Keep the MNLI head and train verified entailment/contradiction/neutral pairs. `uncertain hostility` is not automatically the NLI neutral class, and a false hypothesis is not always contradicted. Candidate pairs from the same message must stay in the same split. Pair expansion does not create more independent conversations. This route needs additional pair-label review and is not the first recommendation.

Suggested order: compact direct classifier first, then BART direct-classifier/LoRA comparison if the measured quality justifies its inference cost. Hardware memory, MPS/GPU compatibility and latency need measurement before choosing a full BART run. No package installation or hardware claim is implied by this plan.

## 6. Evaluation and acceptance decisions

- Report macro-F1 and precision/recall/F1 for all three classes, plus binary hostile/non-hostile metrics and a confusion matrix. Report sample counts for every slice.
- Inspect implicit-hostility recall and false positives on constructive criticism, neutral profanity, self/game frustration and normal combat language.
- Evaluate with/without context and with/without profanity. Later add human-checked profanity removal/insertion pairs that preserve meaning. Keep each pair with its parent in one split.
- Preserve natural prevalence in an independent evaluation set; the stratified pilot cannot estimate deployment precision. Report calibration and uncertainty coverage separately. Fit temperature/thresholds on validation only, with a policy for low-confidence abstention; low softmax confidence is not itself a learned uncertain class.
- Report per-game/source results and a held-out-game test when other games are available. Dota-only results cannot support an any-game claim.
- Measure CPU/MPS/GPU inference latency on the intended machine, including tokenization and context handling; keep STT latency separate from classifier latency. Choose false-positive/false-negative tolerances with the user before setting a release threshold.
- Never claim improvement from one seed or train/test leakage. Use grouped uncertainty estimates where the sample size supports them and publish limitations where it does not.

## Deliverables for a later implementation

Reviewed annotation export and split manifest; immutable baseline predictions; training config and new training entry point; tokenizer and project checkpoint/adapter; metrics and error slices; model card with source, label, context and cross-game limitations. Keep `bart.py` as the baseline throughout. No training or BART code changes are part of the current task.
