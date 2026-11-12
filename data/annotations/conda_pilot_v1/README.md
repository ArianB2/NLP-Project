# CONDA annotation pilot v1 — 240 real source messages

Created November 11, 2026 using contract 0.1.0. These are **model-assisted proposed annotations**, not independent human-reviewed ground truth. No BART model produced these annotations. Each selected message and its available earlier context was inspected to author a per-message decision and rationale in `decisions.tsv`.

## Open these files first

- `sentences_labels.csv`: one sentence and its proposed `label` per row, with `expression`, `category`, target, profanity, context and rationale alongside it. `sentence` preserves the complete source utterance, including [SEPA] merged-message markers; it is not artificially split into grammatical sentences.
- `annotations.jsonl`: the same 240 examples in full data-contract format, retaining original intent and slot labels as provenance.
- `summary.json`: label counts and validation results.

## Coverage

| Group | Number |
| --- | ---: |
| Source E / explicit toxicity | 80 |
| Source I / implicit toxicity | 80 |
| Source O / other | 60 |
| Source A / action | 20 |
| **Total real source messages** | **240** |

After applying our contract rather than copying CONDA's source labels:

| Proposed category | Number |
| --- | ---: |
| Explicit hostility | 61 |
| Implicit hostility | 34 |
| Non-hostile | 79 |
| Uncertain | 66 |

The primary contract labels are `hostile` (95), `non_hostile` (79), and `uncertain` (66). `explicit` and `implicit` remain separate expression annotations; `category` is a convenient derived display column. Neutral here means non-hostile messages, not a list of supposedly neutral individual words.

Of the 240 records, 145 include earlier context and 219 distinct matches are represented. Source-explicit/implicit strata were deliberately oversampled. These counts do not estimate the real-world frequency of hostility, and source-I is not automatically project-implicit.

## Sampling and context

`sample_manifest.json` records the source file hash, seed 20261111, exact IDs, strata, and selection rules. Candidates were sorted by numeric Id. Blank messages and repeated whitespace/case-normalized text were excluded from candidate pools, with source priority E, I, O, A. A seeded Python RNG sampled each pool. This favors diversity and changes prevalence; it is a pilot, not a representative evaluation set.

`selected_source_rows.json` preserves selected CSV rows and up to five available rows from the same match/conversation with strictly earlier `chatTime`. Same-time messages are excluded because order cannot be established. A tie-break by Id only orders context messages already known to precede the target. No future messages or their labels enter the context.

The training split does not provide guaranteed complete conversations. Records with available history say `partial`; records without history say `unknown`, not complete or confirmed first-turn. Original spelling, punctuation, Unicode and game-specific language are preserved. Source file equality is checked before rebuilding.

## Annotation policy and limits

- Labels cover communicated hostility toward people, not negative sentiment or profanity by itself. Direct competence insults are explicit; insinuations, score-based ridicule and contextual sarcasm can be implicit.
- Reporting, commanding in-game actions, constructive disagreement, self-criticism, and complaints about game mechanics are not automatically hostile. Reporting can be hostile when evidence establishes a malicious personal attack, rather than merely because earlier speakers were abusive.
- Easy-win slang often lacks enough context to distinguish celebration from humiliation. Such examples remain uncertain. The 66 uncertain cases are intentional annotations, not missing labels.
- Literal praise is not assumed sarcastic without evidence. Mere hypothetical alternative readings do not make every positive utterance uncertain.
- Profanity means recognizable swear/expletive or obscene sexual wording, including clear abbreviations/obfuscations. Insults, slurs and violent vocabulary are assessed separately; `profanity=absent` does not mean harmless. Unclear multilingual/obfuscated wording receives unknown. This operational distinction needs human review before wider annotation.
- Non-English examples are retained for review rather than translated or silently discarded. Language assignments are tentative; language-qualified reviewers should check them and mixed-language cases.
- Evidence spans currently cover the entire hostile current message. They are valid message-level supports, not carefully localized token gold labels. Read the rationale and context for implicit cases.
- Threat/harm labels describe language, not credibility, imminent danger, or a speaker's actual intentions. In-game killing is not automatically a real-world threat.
- No repeated harassment is inferred from a single insult. No severity score is invented.

Every record has annotation status `proposed`, method `model_assisted`, and quality decision `pending`. The author identifier is software provenance, not a human annotator. **Zero records currently meet the contract's reviewed/adjudicated training-export rule.** There are 174 non-uncertain candidates for human review; the 66 uncertain cases require contextual/language review or exclusion from a binary export.

Do not use the sentence/label file as an approved training split or report benchmark performance against it as human gold. All records come from the official training file and must stay out of any final independent test set. A later group/deduplication audit must include context messages and all edited variants, not just these target rows.

## Rebuild and checks

From the project root:

```sh
env/bin/python data/annotations/conda_pilot_v1/build_pilot.py
```

The builder recreates CSV/JSONL/summary outputs from the saved selection and authored decisions. It checks source hashes, exact CSV rows, 240 distinct IDs, schema fields/enums/types, annotation consistency, evidence bounds, and strictly earlier same-conversation context. Its structural validator supports exactly the JSON Schema keywords currently present and rejects unknown keywords; it is not a general-purpose JSON Schema implementation. It also checks the historical frozen BART snapshot hash. If the live `Bart-mnli/bart.py` differs, it reports that difference without modifying the file or rejecting annotation validation.

These checks verify structure and traceability, not annotation correctness or inter-annotator agreement. Independent human annotation and adjudication remain necessary.
