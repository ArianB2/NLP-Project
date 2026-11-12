# Hostile intent data contract — draft v0.1

This is a proposed shared format for future dataset adapters and annotation. No existing data has been filtered, relabeled, masked, or removed. Scope: insults, humiliation, harassment, identity attacks, threats, and encouragement of harm in game conversations. These are project definitions to review, rather than claims that every source dataset uses the same definitions.

## Unit and task

One record represents one current message, with any available **earlier** messages as context. The task is to label hostility communicated toward a person or group, using the available evidence. We cannot establish a speaker's private mental state from text.

Keep typed messages and STT transcripts in the same format. Audio-derived features are optional and must remain missing for text-only datasets. We should first build a text classifier that can run on either input.

## Primary label

| `hostility` | Definition |
| --- | --- |
| `hostile` | Communicates an attack on a person or group: insult, humiliation, harassment, identity attack, threat, or encouragement of harm |
| `non_hostile` | Available text/context supports ordinary gameplay, constructive criticism, frustration not attacking people, or clearly friendly banter |
| `uncertain` | Cannot reliably distinguish hostility from another reading: ambiguous sarcasm/banter, necessary missing context, or unreliable transcription |
| `null` | Not annotated yet; this is different from an annotator deciding that the example is uncertain |

Profanity, slang, disagreement, a negative sentiment, or mentioning a violent game action is insufficient by itself. Hostility can occur without any of these. Friendly intent must be supported by the conversation, rather than assumed because an insult occurs during a game. Quoting abuse to report it is not automatically endorsing it.

`uncertain` is a retained annotation outcome. Exclude it from a first binary training export and report how much was excluded; do not silently turn it into `non_hostile`. A model's runtime abstention threshold is a later decision, separate from these annotation labels.

## Separate annotation dimensions

| Field | Values | Rule |
| --- | --- | --- |
| `expression` | `explicit`, `implicit`, `not_applicable`, `unknown` | Explicit: attack conveyed directly. Implicit: attack inferred through sarcasm, insinuation, or context. Use not_applicable for non-hostile and unknown for uncertain |
| `target` | `individual`, `team`, `group`, `self`, `none`, `unknown` | Who is attacked, not simply who receives the message. Self-directed frustration is not interpersonal hostility under this contract |
| `behaviors` | Any of `insult`, `humiliation`, `harassment`, `identity_attack`, `threat`, `encouragement_of_harm`; or `null` | Multiple values allowed. `[]` means assessed and none present; null means unassessed/unknown. Repeated harassment requires evidence across messages |
| `profanity` | `present`, `absent`, `unknown` | Independent of hostility. Preserve this dimension to evaluate hostility with and without profanity |
| `context_required` | `true`, `false`, `null` | Whether prior context is needed to justify this annotation; null if not assessed |
| `evidence` | List of spans, or `null` | Optional explanatory spans. Empty list means assessed with no localizable span; null means unannotated |
| `rationale` | String or null | Brief explanation, especially for implicit or uncertain cases; annotation metadata, never model input |

Do not add a severity score initially. First agree on the primary boundary and verify annotator agreement. Slang/game-term spans can be preserved from a source or added as optional evidence; they are not required classification targets.

## Shared record fields

Store records as UTF-8 JSONL: one JSON object per line. See `data_contract.schema.json` and `data_contract.examples.jsonl`.

| Field | Meaning |
| --- | --- |
| `schema_version` | `0.1.0`; changing annotation meaning requires a version change |
| `id` | Stable namespaced ID, such as `conda:train:11263`; never just a row number |
| `source` | Dataset name, release/version, original ID, source URL, original split, and license identifier/reference when known |
| `source_labels` | Original labels/annotations without reinterpretation; preserve CONDA slots here |
| `game` | Game name/ID when known; null if unknown; never force everything to Dota |
| `modality` | `text_chat` or `stt` |
| `language` | Language code when known, otherwise null; do not infer English solely from the source name |
| `conversation_id` | Namespaced conversation ID when known, otherwise null |
| `leakage_group_id` | Namespaced match/session grouping key for splitting; null if unavailable and requires split review |
| `speaker_id` | Source-local pseudonymous speaker ID when available, otherwise null; avoid display names as model features |
| `text_raw` | Original message/transcript verbatim; never overwrite |
| `text` | Current analysis/model text; initially equal to text_raw |
| `transforms` | Ordered documented transformations from text_raw to text, initially empty |
| `context` | Earlier messages, oldest first, as `{id, speaker_id, text}` objects; never future messages |
| `context_available` | `complete`, `partial`, `none`, or `unknown`; complete means all prior messages in the declared conversation, not necessarily the whole match |
| `annotation` | Project labels plus annotation version, method, status, and annotator IDs |
| `quality` | `pending`, `include`, or `exclude`; reasons and optional parent ID for synthetic/edited examples |

Unknown values are null, not an invented category or a negative finding. An empty context list alone does not distinguish the first message of a conversation from missing history: use context_available. Timestamps and audio confidence are not universally available; they can be introduced with a documented optional extension in a later version.

## Evidence spans and consistency

Evidence uses zero-based Unicode code-point offsets `[start, end)` into the **current** `text` or the specified context message's text. `message_id` identifies that message; `kind` is `hostility`, `profanity`, `slang`, or `game_term`. Recompute spans whenever text changes. Whole-message/context reasoning can justify implicit hostility even without a single hostile word.

- A hostile annotation must have expression explicit/implicit and target individual/team/group/unknown. Unknown target is acceptable only if an attack on people is clear but its recipient cannot be resolved.
- A non-hostile annotation must have expression not_applicable and target none; assessed behaviors must be empty.
- An uncertain annotation must have expression unknown; retain a rationale explaining the uncertainty.
- A pending annotation has hostility null and status pending; do not present imported candidate labels as reviewed annotations.
- `profanity=absent` requires an actual assessment. Missing source tags cannot establish absence.
- Evidence offsets must satisfy `0 <= start < end <= len(referenced_text)` and reference the current message or supplied context.
- Only final STT messages become stable training records. Partial hypotheses must not become duplicate training examples.

These semantic rules need a future dataset validator in addition to the JSON Schema, which checks field structure and allowed values.

## Annotation provenance and review

`annotation.status`: `pending`, `proposed`, `reviewed`, or `adjudicated`.
`annotation.method`: `human`, `source_mapping`, `model_assisted`, or null before annotation.
Record `annotation.guideline_version` and local pseudonymous `annotator_ids`.

Start with an independently double-annotated pilot spanning all available sources, games, profanity conditions, and ambiguous cases. Compare agreement on hostility and explicitness, adjudicate disagreements, and update the guidelines before annotating at scale. Model suggestions and source mappings remain proposed until reviewed. Do not use classifier confidence as human agreement.

For the first supervised export, include only records with quality include, reviewed/adjudicated annotations, and hostility hostile/non_hostile. Export only current text and the permitted earlier context as model input. Labels, rationales, evidence, source labels, split IDs, and review metadata must not enter the text input.

## CONDA adapter policy

The source has human intent labels and separate lexicon-derived token labels. Source E/I can propose hostile candidates, and A/O can propose non-hostile candidates, but this mapping requires review against our definitions. Store every original label in source_labels. T is a lexical toxicity tag, not a profanity or hostility ground truth. C/D/S are not labels to turn into hostility.

Preserve [SEPA] and its meaning when importing merged messages. Preserve raw text and avoid inventing turn timestamps when original sub-message times are unavailable. CONDA's character, Dota-specific, and slang tags can be retained in source_labels without making them required targets for every new dataset. [CONDA annotation and preprocessing](https://arxiv.org/html/2106.06213v2#S3).

## Generalize across games

Use game-independent project labels and retain game as metadata. Do not delete hero names, abilities, slang, or game references from raw text: removing them can change the meaning. Optional generic entity replacements belong in a separate, versioned transformation experiment, keeping source text and parent relationships.

Evaluate on held-out games or sources to test generalization. Removing Dota-specific columns does not prove a model has stopped depending on Dota vocabulary. A non-gaming dataset may help with some behaviors, but it must meet the same contract and its domain should remain identifiable through source metadata.

## Filtering and splitting policy

Keep filtering reversible: mark excluded records with reasons rather than deleting them. Flag blank/unusable text, unsupported languages, missing required context, uncertain transcripts, and duplicates for review. Missing token annotations alone do not make a record unusable for message classification. Repeated short utterances can be legitimate and need contextual review before deduplication.

Preserve official source splits. Audit match/conversation overlap and duplicates across them before evaluation; keep each match/session together for any newly created split. Keep edited variants and duplicates with their originals in the same split, using parent IDs and a deduplication manifest. Never add held-out evaluation examples or their labels to annotation prompts used to generate training data. If history is unavailable, record that limitation instead of reconstructing a supposedly complete conversation from a shuffled training split.

## Synthetic examples

The example JSONL is invented to illustrate the contract, not a training corpus:

- “Please defend the base.” — non-hostile gameplay instruction.
- “You are useless.” — explicit insult/humiliation directed at another player, without profanity.
- “Nice job.” — uncertain when context does not establish praise versus sarcasm.

Next implementation after reviewing this draft: a lossless CONDA adapter and a semantic validator, followed by a small annotation pilot. Dataset discovery and bulk cleaning come after those decisions.
