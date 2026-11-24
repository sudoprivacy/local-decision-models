"""Compare three saved office profiles around a witnessed manual light switch operation."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path


class IncompleteObservation(ValueError):
    pass


def _observation(profile):
    connection = profile["connection_id"]
    observed = datetime.fromisoformat(profile["observed_at"])
    if (
        not re.fullmatch(r"[0-9a-f]{32}", connection)
        or observed.utcoffset() is None
        or profile["source"] != "yeelight_pro_lan_v2_22_inventory"
        or type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
    ):
        raise IncompleteObservation("invalid_profile")
    lamps = {}
    for node in profile["collections"]["nodes"]:
        if type(node["product_type"]) is not int or node["product_type"] != 3:
            continue
        ref = node["ref"]
        state = node["state"]
        power = state["power"]
        brightness = state["brightness_percent"]
        kelvin = state["color_temperature_kelvin"]
        if (
            not re.fullmatch(connection + r"/2/[1-9][0-9]{0,19}", ref)
            or int(ref.rsplit("/", 1)[1]) >= 2**64
            or ref in lamps
            or node["node_type"] != 2
            or node["online"] is not True
            or type(power) is not bool
            or type(brightness) is not int
            or not 1 <= brightness <= 100
            or type(kelvin) is not int
            or not 2700 <= kelvin <= 6500
        ):
            raise IncompleteObservation("incomplete_or_duplicate_lamp")
        lamps[ref] = (power, brightness, kelvin)
    if not lamps:
        raise IncompleteObservation("no_tunable_lamps")
    return connection, observed, lamps


def compare(profiles):
    before, toggled, restored = [_observation(profile) for profile in profiles]
    if len({before[0], toggled[0], restored[0]}) != 1:
        raise IncompleteObservation("different_gateway_registrations")
    if not before[1] < toggled[1] < restored[1]:
        raise IncompleteObservation("reused_or_unordered_snapshots")
    start, middle, end = before[2], toggled[2], restored[2]
    if start.keys() != middle.keys() or start.keys() != end.keys():
        raise IncompleteObservation("lamp_inventory_changed")
    if end != start:
        raise IncompleteObservation("initial_state_not_restored")
    changed = [ref for ref in start if start[ref] != middle[ref]]
    if len(changed) != 1:
        raise IncompleteObservation("expected_one_changed_lamp")
    ref = changed[0]
    if start[ref][0] == middle[ref][0] or start[ref][1:] != middle[ref][1:]:
        raise IncompleteObservation("expected_power_only_change")
    return ref, len(start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Private directory holding the three profile copies")
    directory = parser.parse_args().directory
    try:
        names = ("before.private.json", "toggled.private.json", "restored.private.json")
        samples = [(directory / name).read_bytes() for name in names]
        ref, count = compare([json.loads(sample) for sample in samples])
        digests = [hashlib.sha256(sample).hexdigest() for sample in samples]
        receipt = json.dumps({"snapshot_sha256": digests, "candidate_ref": ref}, indent=2) + "\n"
        name = "candidate-" + hashlib.sha256(receipt.encode()).hexdigest() + ".private.json"
        fd, temporary = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
                file.write(receipt)
            os.replace(temporary, directory / name)
        finally:
            Path(temporary).unlink(missing_ok=True)
        result = {"status": "candidate_for_visual_confirmation", "tunable_lamps": count, "receipt": name}
    except (OSError, KeyError, ValueError, TypeError, OverflowError) as error:
        code = str(error) if isinstance(error, IncompleteObservation) else "invalid_or_unavailable_profile"
        print(json.dumps({"status": "incomplete", "reason": code, "physical_commands": 0}))
        return 1
    print(json.dumps({**result, "physical_commands": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
