# Session notes

## Run 1 — November 9, 2026 — Mac microphone to OpenAI STT

- Created `requirements.txt` first, then inspected the project. Found an existing `env` Python environment, an empty `main.py`, and a small `openai_service.py` stub.
- Read OpenAI's [Realtime transcription documentation](https://developers.openai.com/api/docs/guides/realtime-transcription) and [WebSocket documentation](https://developers.openai.com/api/docs/guides/voice-websockets?api=realtime).
- Ran `env/bin/python -m pip install -r requirements.txt`. All needed libraries were already installed; pinned their installed versions in `requirements.txt`: numpy, python-dotenv, sounddevice, and websockets. The direct WebSocket implementation does not need the OpenAI SDK.
- Implemented `openai_service.py`: loads `OPENAI_API_KEY` from the project's `.env`, connects and configures a transcription session, streams audio, finalizes turns, handles API errors, and collects partial/final transcripts by item ID. Final text follows speech order even if replies complete out of order. Transcripts stay in memory and are printed; no recording or transcript file is saved.
- Added `mic.py`: starts/stops the Mac microphone through sounddevice and native Core Audio. Captures mono PCM16 audio at 24 kHz in 20 ms frames. Passes frames safely to asyncio and reports input/queue errors.
- Added `record_STT.py`: connects the microphone and service, prints live partial/final text, and stops with Ctrl+C. On stop, sends remaining speech and waits up to 15 seconds for final transcripts before closing.
- Configured `gpt-live-transcribe` with low delay. Simple local energy detection keeps 200 ms of pre-roll and commits a turn after 500 ms of silence or 30 seconds of continuous audio. Tune `--threshold` for quiet voices or noisy rooms; actual latency needs live microphone testing.
- Added `test_stt.py`: five offline tests passed for session setup/cleanup, PCM encoding and short-turn padding, partial text and out-of-order finals, API errors, silence/pre-roll and stop flushing, and missing-key validation.
- Verified CLI help and `git diff --check`. Updated `.gitignore` to ignore Python bytecode.
- Live connection check reached OpenAI, but the API returned: "You have no credits remaining." Full microphone-to-transcript testing and latency measurement remain pending until API credits are available. No microphone audio was sent in the connection-only check.

### Run

- Put a valid key in `.env`: `OPENAI_API_KEY=your_key_here`. The existing shell environment takes precedence if the variable is already set. Never commit the key.
- Install dependencies: `env/bin/python -m pip install -r requirements.txt`.
- List microphones: `env/bin/python record_STT.py --list-devices`.
- Start transcription: `env/bin/python record_STT.py`. Speak, then press Ctrl+C to finish.
- Optional settings: `env/bin/python record_STT.py --device 0 --languages en --delay minimal --threshold 300 --silence-ms 400`.
- Allow microphone access for the terminal or IDE in macOS System Settings > Privacy & Security > Microphone if requested.
- Run offline checks: `env/bin/python -m unittest -v test_stt.py`.

## Run 2 — November 9, 2026 — Retry after adding credits

- Reran the connection-only check using `OpenAITranscriptionSession` in `openai_service.py`.
- OpenAI successfully accepted the authenticated transcription session and its configuration; the connection closed cleanly.
- No microphone audio was recorded or sent. Live speech transcription and latency measurement still need a microphone test.

## Run 3 — November 9, 2026 — Simple start command

- Added `start`: run `./start` from the project folder to launch the microphone and STT using the existing `env` environment. No environment activation is required. Options such as `./start --list-devices` still work.
- Added the entry point in `main.py`, which runs the pipeline from `record_STT.py`.
- Updated the startup message in `record_STT.py`. Partial and final transcriptions print in the terminal as they arrive; Ctrl+C stops recording and prints the complete session transcript.

## Run 4 — November 9, 2026 — Understand CONDA

- Installed pandas and matplotlib (which provides `matplotlib.pyplot`) in `env`; NumPy was already installed. Updated `requirements.txt` with installed versions. `pip check` found no broken dependencies.
- Read the CONDA paper and repository to confirm the message and token label definitions.
- Added and ran `analyze_conda.py` against `CONDA_train.csv`, preserving literal chat strings such as NA.
- Generated `analysis/report.md`, `analysis/overview.png`, `analysis/overview.pdf`, and three CSV summary tables. The report explains all columns, labels, class balance, missing values, context grouping, and implications for future STT classification.
- Found 26,921 messages: 19,982 O, 3,528 E, 1,692 I, and 1,719 A. E + I accounts for 19.39%. There are 834 rows with no token annotations and 7 blank utterances. Separated SEPA markers from the six semantic token classes.
- Reproduce with `env/bin/python analyze_conda.py`. This run analyzed data without training a model or changing the microphone pipeline.

## Run 5 — November 9, 2026 — Common annotation contract draft

- Drafted `DATA_CONTRACT.md` for game-independent hostile intent classification before cleaning or importing more datasets. Scope includes insults, humiliation, harassment, and threats, as requested.
- Separated the primary hostility decision from explicitness, target, behavior, profanity, context, and optional evidence spans. Distinguished uncertain annotations from records awaiting annotation.
- Added `data_contract.schema.json` for the shared JSONL structure and `data_contract.examples.jsonl` with three synthetic illustrations explicitly excluded from training.
- Documented source provenance, raw-text preservation, CONDA mapping review, earlier-context rules, reversible filtering, and split leakage prevention. Preserved all existing source data; no model was trained and no source annotations were changed.
- The contract remains a draft for review. A lossless dataset adapter, semantic validator, and annotation pilot are subsequent implementation steps.

## Run 6 — November 11, 2026 — CONDA annotation pilot and BART plan

- Inspected `DATA_CONTRACT.md`, the schema, and `Bart-mnli/bart.py`. Preserved existing user edits and left `bart.py` unchanged.
- Sampled 240 distinct nonblank CONDA messages with seed 20261111: 80 source E, 80 source I, 60 source O and 20 source A. Saved source IDs, the CSV hash, selection rules and available strictly earlier context in `data/annotations/conda_pilot_v1/`.
- Authored individual model-assisted proposed annotations and rationales against our contract, rather than blindly copying source labels. Result: 61 explicit-hostile, 34 implicit-hostile, 79 non-hostile and 66 uncertain messages. Ambiguous easy-win slang and multilingual fragments remain uncertain where appropriate.
- Created `sentences_labels.csv` for sentence/label inspection and `annotations.jsonl` for complete contract records, plus a README and summary. All annotations are proposed and pending human review; no records were represented as human-reviewed or exported as approved training data.
- Added `build_pilot.py` to regenerate exports and check all 240 records for source equality, schema structure, label consistency, evidence bounds and earlier-only context. The script also verifies the unchanged BART source hash.
- Froze the exact BART hypotheses, template, original example messages, source snapshot/hash and package inventory under `Bart-mnli/baselines/2026-11-11/`. Model weights/revision were not downloaded or modified; the original source did not pin a model revision.
- Wrote `Bart-mnli/TRAINING_PLAN.md` using official Hugging Face documentation. The plan covers a compact pretrained classifier, staged weight unfreezing, a BART/LoRA alternative, human annotation review, grouped splits, context and profanity evaluation, and cross-game/STT tests. This run did not implement training, run BART inference or change `bart.py`.

## Run 7 — November 12, 2026 — Transfer to the correct working directory

- Corrected the active project location to `/Volumes/Samsung T7/HIC`. Prior work had been written under `/Users/wadoodabdul/dev/HIC`.
- Compared prior implementation files, requirements, contract and analysis outputs; these already matched the external-drive project and were left in place.
- Copied the 240-example annotation pilot, historical BART snapshot/configuration/package inventory, training plan and missing session history to this project.
- Preserved the external-drive `Bart-mnli/bart.py`, whose six-label hypotheses differ from the historical five-label dev-directory baseline. Added a provenance note to the plan.
- Adjusted the transferred annotation builder to verify the historical snapshot itself and report differences from the live BART file without rejecting annotation validation. Updated the pilot README accordingly.
- Did not replace `.env`, the Python environment, Git metadata, source CSV, or existing BART code. Use this external-drive directory for subsequent project work.
