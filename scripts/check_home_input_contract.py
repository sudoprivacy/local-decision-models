"""Check the real MS-SWIFT train/inference token contract without loading weights."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    import home_swift_plugin as plugin
    from swift.dataset.preprocessor.decision import JevPreprocessor
    from swift.model import get_model_processor
    from swift.template import get_template

    _, processor = get_model_processor(
        args.model, model_type=plugin.MODEL_TYPE, task_type="decision", load_model=False
    )
    template = get_template(processor, max_length=2048, truncation_strategy="raise")
    full_template = get_template(processor, max_length=None)
    preprocessor = plugin.HomeDecisionPreprocessor()
    rows = json.loads((args.bundle / "records.json").read_text(encoding="utf-8"))
    assert rows, "empty_bundle"
    lengths, dropped_state_controls, sensitivity_checks = [], 0, 0
    for row in rows:
        raw = copy.deepcopy(row["model_input"])
        raw["questions"][0]["gold"] = row["annotation"]["gold_option_index"]
        training = preprocessor.preprocess(copy.deepcopy(raw))[0]
        state, question = raw["state"], raw["questions"][0]
        inference = plugin.home_decision_request(state, question)
        assert json.loads(training["messages"][0]["content"]) == state
        assert {k: v for k, v in training.items() if k != "target_probs"} == inference
        template.set_mode("train")
        encoded_train = template.encode(copy.deepcopy(training))
        template.set_mode("transformers")
        encoded_infer = template.encode(copy.deepcopy(inference))
        full_template.set_mode("transformers")
        encoded_full = full_template.encode(copy.deepcopy(inference))
        assert encoded_train["input_ids"] == encoded_infer["input_ids"] == encoded_full["input_ids"]
        assert encoded_train["decision_meta"] == encoded_infer["decision_meta"]
        assert 0 < len(encoded_train["input_ids"]) <= 2048
        lengths.append(len(encoded_train["input_ids"]))
        changed_gold = copy.deepcopy(raw)
        changed_gold["questions"][0]["gold"] = (question["gold"] + 1) % len(question["options"])
        template.set_mode("train")
        changed_encoded = template.encode(preprocessor.preprocess(changed_gold)[0])
        assert changed_encoded["input_ids"] == encoded_train["input_ids"], "gold_leaks_into_input"
        legacy = JevPreprocessor().preprocess(copy.deepcopy(raw))[0]
        dropped_state_controls += legacy["messages"][0]["content"] != inference["messages"][0]["content"]
        for field in ("online", "state_fresh", "power", "preset", "presence"):
            changed = copy.deepcopy(state)
            if field in ("online", "state_fresh"):
                changed["devices"][0][field] = not changed["devices"][0][field]
            elif field == "power":
                changed["devices"][0]["properties"]["p"] = not changed["devices"][0]["properties"]["p"]
            elif field == "preset":
                changed["home_profile"]["reading_preset"]["l"] += 1
            else:
                changed["devices"][1]["presence"] = not changed["devices"][1]["presence"]
            template.set_mode("transformers")
            encoded_changed = template.encode(plugin.home_decision_request(changed, question))
            assert encoded_changed["input_ids"] != encoded_infer["input_ids"], "nested_state_disappeared"
            sensitivity_checks += 1
    assert dropped_state_controls == len(rows), "legacy_negative_control_not_reproduced"
    receipt = {
        "status": "passed",
        "weights_loaded": False,
        "rows": len(rows),
        "train_inference_token_matches": len(rows),
        "gold_independence_checks": len(rows),
        "nested_state_sensitivity_checks": sensitivity_checks,
        "legacy_preprocessor_mismatches": dropped_state_controls,
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "truncated_rows": 0,
        "bundle_sha256": hashlib.sha256((args.bundle / "bundle.json").read_bytes()).hexdigest(),
        "physical_commands_sent": 0,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
