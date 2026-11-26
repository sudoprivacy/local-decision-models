"""Score an existing synthetic turn-on request, read the mock lamp and reconsider once."""

import argparse
import json
import math
from pathlib import Path

from home_target_score import digest, load_engine, score_options

from ldm.data import atomic_json
from ldm.home_mock import MockGateway
from ldm.home_refresh import MockObservation, review_lamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "adapter", "bundle", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("choose a private output path outside the checkout")
    manifest = json.loads((args.bundle / "bundle.json").read_bytes())
    if digest(args.bundle / "records.json") != manifest["records_sha256"]:
        parser.error("bundle integrity check failed")
    (row,) = [r for r in json.loads((args.bundle / "records.json").read_bytes()) if r["id"] == args.row_id]
    state = row["model_input"]["state"]
    (question,) = row["model_input"]["questions"]
    options = [a["id"] for a in row["candidate_actions"]]
    if len(set(options)) != 8 or [a["description"] for a in row["candidate_actions"]] != question["options"]:
        parser.error("candidate descriptions must match the scoring order")
    engine = load_engine(args.model, args.adapter)
    cases = {}
    for name, power, verified, online in [
        ("already_on", True, True, True),
        ("still_off", False, True, True),
        ("cached_only", True, False, True),
        ("offline", False, False, False),
    ]:
        forwards = []

        def choose(s):
            result = score_options(engine, s, question)
            v = result["probabilities"]
            if not (
                len(v) == len(options)
                and all(math.isfinite(x) and 0 <= x <= 1 for x in v)
                and abs(sum(v) - 1) < 1e-4
                and v.count(max(v)) == 1
            ):
                raise ValueError("invalid model scores")
            forwards.append({"state": s, **result})
            return options[v.index(max(v))]

        with MockGateway() as gateway:
            # Set up the simulated device independently of the cached model input.
            with gateway.lock:
                gateway.nodes[0]["params"]["p"] = power
            gateway.set_online(online)

            def observe():
                return MockObservation(gateway.reader().query("gateway_get.node", 101), verified)

            trace = review_lamp(state, choose, observe)
            if gateway.mock_commands_sent:
                raise RuntimeError("unexpected mock write")
        cases[name] = {**trace, "forwards": forwards}
    atomic_json(
        args.output,
        {
            "adapter_files": {
                n: digest(args.adapter / n)
                for n in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "scoring_head.json",
                    "scoring_head.safetensors",
                )
            },
            "row_id": args.row_id,
            "cases": cases,
            "physical_commands_sent": 0,
            "mock_commands_sent": 0,
        },
    )
    print(json.dumps({k: {f: v for f, v in c.items() if f != "forwards"} for k, c in cases.items()}))


if __name__ == "__main__":
    main()
