import json
import subprocess
import sys
from pathlib import Path

import pytest

from ldm.home_profile import build_home_profile


@pytest.mark.parametrize(
    ("situation", "expected"),
    [
        ("one_light", "candidate_for_visual_confirmation"),
        ("linked", "expected_one_changed_lamp"),
        ("no_change", "expected_one_changed_lamp"),
        ("not_restored", "initial_state_not_restored"),
        ("brightness", "expected_power_only_change"),
        ("offline", "incomplete_or_duplicate_lamp"),
        ("different_gateway", "different_gateway_registrations"),
        ("missing_lamp", "lamp_inventory_changed"),
        ("reused_snapshot", "reused_or_unordered_snapshots"),
    ],
)
def test_manual_switch_comparison(tmp_path, situation, expected):
    names = ("before", "toggled", "restored")
    for i, name in enumerate(names):
        nodes = [
            {"id": 100 + index, "pt": 3, "o": True, "params": {"p": False, "l": 40, "ct": 3500}}
            for index in range(4)
        ]
        gateway, seconds = "a" * 32, i
        if name == "toggled":
            nodes[0]["params"]["p"] = situation != "no_change"
            if situation == "linked":
                nodes[1]["params"]["p"] = True
            elif situation == "offline":
                nodes[1]["o"] = False
            elif situation == "brightness":
                nodes[0]["params"]["l"] = 50
            elif situation == "missing_lamp":
                nodes.pop()
            elif situation == "different_gateway":
                gateway = "b" * 32
            elif situation == "reused_snapshot":
                seconds = 0
        if name == "restored" and situation == "not_restored":
            nodes[0]["params"]["p"] = True
        profile = build_home_profile({"nodes": nodes}, gateway, f"2026-11-24T00:00:0{seconds}Z")
        (tmp_path / f"{name}.private.json").write_text(json.dumps(profile), encoding="utf-8")

    tool = Path(__file__).resolve().parents[1] / "scripts" / "compare_lamp_observations.py"
    run = subprocess.run([sys.executable, str(tool), str(tmp_path)], capture_output=True, text=True)
    result = json.loads(run.stdout)
    assert result["physical_commands"] == 0
    assert "a" * 32 not in run.stdout
    if situation == "one_light":
        assert run.returncode == 0 and result["status"] == expected and result["tunable_lamps"] == 4
        receipt = json.loads((tmp_path / result["receipt"]).read_bytes())
        assert receipt["candidate_ref"] == "a" * 32 + "/2/100"
        assert len(set(receipt["snapshot_sha256"])) == 3
    else:
        assert run.returncode == 1 and result["reason"] == expected
        assert not list(tmp_path.glob("candidate-*.private.json"))
