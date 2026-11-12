"""Rebuild and validate exports from saved source rows and per-message decisions.

Run: env/bin/python data/annotations/conda_pilot_v1/build_pilot.py
No model inference or training; decisions.tsv contains the authored annotations.
"""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def check_schema(value, schema, root, path="record"):
    """Validate exactly the JSON Schema keyword subset used by our contract.

    Not a general-purpose Draft 2020-12 implementation. Reject unknown keywords
    so a future schema extension cannot silently bypass structural validation.
    """
    allowed = {"$schema", "title", "$defs", "$ref", "type", "required", "properties",
               "additionalProperties", "const", "enum", "minLength", "minimum",
               "items", "uniqueItems"}
    assert not set(schema) - allowed, f"Unsupported schema keyword at {path}"
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        check_schema(value, target, root, path)
    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        tests = {"object": type(value) is dict, "array": type(value) is list,
                 "string": type(value) is str, "integer": type(value) is int,
                 "boolean": type(value) is bool, "null": value is None}
        assert any(tests[t] for t in types), f"Wrong type at {path}"
    if "enum" in schema:
        assert value in schema["enum"], f"Invalid enum at {path}"
    if "const" in schema:
        assert value == schema["const"], f"Invalid const at {path}"
    if isinstance(value, dict):
        assert set(schema.get("required", [])) <= value.keys(), f"Missing keys at {path}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            assert value.keys() <= properties.keys(), f"Extra keys at {path}"
        for key, subschema in properties.items():
            if key in value:
                check_schema(value[key], subschema, root, f"{path}.{key}")
    if isinstance(value, list):
        if schema.get("uniqueItems"):
            assert len({json.dumps(x, sort_keys=True) for x in value}) == len(value), path
        if "items" in schema:
            for i, item in enumerate(value):
                check_schema(item, schema["items"], root, f"{path}[{i}]")
    if isinstance(value, str) and "minLength" in schema:
        assert len(value) >= schema["minLength"], path
    if type(value) in (int, float) and "minimum" in schema:
        assert value >= schema["minimum"], path


def main():
    manifest = json.loads((HERE / "sample_manifest.json").read_text())
    source_bytes = (ROOT / manifest["source_file"]).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == manifest["source_sha256"], "Source CSV changed"
    source = {r["Id"]: r for r in csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"), newline=""))}
    sample = json.loads((HERE / "selected_source_rows.json").read_text())
    with (HERE / "decisions.tsv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle, delimiter="\t"))
    assert len(sample) == len(decisions) == 240
    assert [x["row"]["Id"] for x in sample] == manifest["ids"]
    assert len(set(manifest["ids"])) == 240
    schema = json.loads((ROOT / "data_contract.schema.json").read_text())
    records, flat = [], []
    categories = {"E": ("hostile", "explicit"), "I": ("hostile", "implicit"),
                  "N": ("non_hostile", "not_applicable"), "U": ("uncertain", "unknown")}
    languages = {"34455": "tl", "33956": "ru", "40585": "tl", "28151": "pt"}
    unknown_language = {"17392", "2064", "20547", "26062", "37930", "4682", "32839", "44022"}
    for number, (item, decision) in enumerate(zip(sample, decisions), 1):
        assert int(decision["n"]) == number
        row = item["row"]
        assert row == source[row["Id"]], f"Source mismatch for {row['Id']}"
        message_id = f"conda:train:{row['Id']}"
        match = f"conda:match:{row['matchId']}"
        speaker = lambda r: f"{match}:slot:{r['playerSlot']}"
        context = []
        previous_time = float("-inf")
        for prior in item["previous_rows"]:
            assert prior == source[prior["Id"]]
            assert (prior["matchId"], prior["conversationId"]) == (row["matchId"], row["conversationId"])
            assert previous_time <= float(prior["chatTime"]) < float(row["chatTime"]), "Future/tied context"
            previous_time = float(prior["chatTime"])
            context.append({"id": f"conda:train:{prior['Id']}", "speaker_id": speaker(prior), "text": prior["utterance"]})
        hostility, expression = categories[decision["category"]]
        behaviors = decision["behaviors"]
        behaviors = None if behaviors == "?" else ([] if behaviors == "-" else behaviors.split(","))
        evidence = [{"message_id": message_id, "start": 0, "end": len(row["utterance"]), "kind": "hostility"}] if hostility == "hostile" else ([] if hostility == "non_hostile" else None)
        reasons = ["Model-assisted proposal; independent human review required before supervised export"]
        if hostility == "uncertain":
            reasons.append("Uncertain cases excluded from a future binary training export unless resolved")
        record = {
            "schema_version": "0.1.0", "id": message_id,
            "source": {"dataset": "CONDA", "version": f"local-sha256:{manifest['source_sha256']}",
                       "original_id": row["Id"], "url": "https://github.com/usydnlp/CONDA",
                       "split": "train", "license": None},
            "source_labels": {k: row[k] for k in ["intentClass", "slotClasses", "slotTokens", "matchId", "conversationId", "chatTime", "playerSlot"]},
            "game": "dota_2", "modality": "text_chat",
            "language": languages.get(row["Id"], None if row["Id"] in unknown_language else "en"),
            "conversation_id": f"{match}:conversation:{row['conversationId']}",
            "leakage_group_id": match, "speaker_id": speaker(row),
            "text_raw": row["utterance"], "text": row["utterance"], "transforms": [],
            "context": context, "context_available": "partial" if context else "unknown",
            "annotation": {"hostility": hostility, "expression": expression, "target": decision["target"],
                           "behaviors": behaviors, "profanity": decision["profanity"],
                           "context_required": decision["context_required"] == "true",
                           "evidence": evidence, "rationale": decision["rationale"],
                           "status": "proposed", "method": "model_assisted",
                           "guideline_version": "0.1.0", "annotator_ids": ["codex:pilot-2026-11-11"]},
            "quality": {"decision": "pending", "reasons": reasons, "parent_id": None},
        }
        check_schema(record, schema, schema)
        annotation = record["annotation"]
        assert annotation["rationale"]
        if hostility == "non_hostile":
            assert expression == "not_applicable" and annotation["target"] == "none" and behaviors == []
        if hostility == "hostile":
            assert expression in ("explicit", "implicit")
            assert annotation["target"] in ("individual", "team", "group", "unknown")
            assert behaviors
        if hostility == "uncertain":
            assert expression == "unknown"
        texts = {message_id: record["text"], **{c["id"]: c["text"] for c in context}}
        for span in evidence or []:
            assert 0 <= span["start"] < span["end"] <= len(texts[span["message_id"]])
        records.append(record)
        flat.append({"id": message_id, "sentence": record["text"], "label": hostility,
                     "expression": expression,
                     "category": {"E": "explicit_hostility", "I": "implicit_hostility", "N": "non_hostile", "U": "uncertain"}[decision["category"]],
                     "target": annotation["target"], "behaviors": json.dumps(behaviors),
                     "profanity": annotation["profanity"], "language": record["language"],
                     "context_required": annotation["context_required"],
                     "context": json.dumps(context, ensure_ascii=False), "context_available": record["context_available"],
                     "source_intent": row["intentClass"], "match_id": record["leakage_group_id"],
                     "rationale": annotation["rationale"], "annotation_status": "proposed", "quality_decision": "pending"})
    (HERE / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    with (HERE / "sentences_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {
        "total": len(records), "source_strata": dict(Counter(r["source_labels"]["intentClass"] for r in records)),
        "categories": dict(Counter(r["category"] for r in flat)),
        "hostility": dict(Counter(r["label"] for r in flat)),
        "profanity": dict(Counter(r["profanity"] for r in flat)),
        "with_earlier_context": sum(bool(r["context"]) for r in records),
        "human_reviewed": 0, "currently_training_eligible_under_contract": 0,
        "non_uncertain_candidates_for_review": sum(r["label"] != "uncertain" for r in flat),
        "unique_matches": len({r["leakage_group_id"] for r in records}),
        "checks": ["240 unique source IDs", "source file hash and original row equality",
                   "contract structural keyword-subset validation", "annotation semantic consistency",
                   "evidence bounds", "strictly earlier same-conversation context"],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    baseline = ROOT / "Bart-mnli/baselines/2026-11-11"
    frozen = json.loads((baseline / "hypothesis.json").read_text())
    assert hashlib.sha256((baseline / "bart.py.snapshot").read_bytes()).hexdigest() == frozen["source_sha256"], "Frozen snapshot changed"
    if hashlib.sha256((ROOT / "Bart-mnli/bart.py").read_bytes()).hexdigest() != frozen["source_sha256"]:
        print("Note: current bart.py differs from the historical dev-directory baseline; current file preserved.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
