# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["pydantic==2.14.3", "psutil==7.2.2"]
# ///
"""Lightweight read-only entry point; no training stack or cloud credentials."""

import argparse
import json
import platform
import sys
from pathlib import Path


def self_test() -> dict:
    """Use only an owned loopback gateway; never discover or contact the LAN."""
    from ldm.contracts import DomainError
    from ldm.home_mock import MockGateway
    from ldm.home_profile import build_home_profile, profile_summary

    with MockGateway() as gateway:
        reader = gateway.reader()
        topology = reader.query("gateway_get.topology")
        inventory = reader.inventory()
        expected = {"nodes": 2, "rooms": 1, "groups": 1, "scenes": 1}
        if {key: len(rows) for key, rows in inventory.items()} != expected or not topology:
            raise DomainError("lan_self_test_inventory_mismatch")
        try:
            reader.query("gateway_set.prop")
        except DomainError as error:
            if error.code != "gateway_read_only":
                raise
        else:
            raise DomainError("lan_self_test_write_guard_failed")
        if gateway.mock_commands_sent:
            raise DomainError("lan_self_test_unexpected_write")
        profile = build_home_profile(inventory, "0" * 32, "2026-11-19T00:00:00+00:00")
    training_imported = any(name in sys.modules for name in ("torch", "transformers", "peft"))
    if training_imported:
        raise DomainError("lan_self_test_training_dependency_loaded")
    return {
        "status": "passed",
        "scope": "synthetic_loopback_only",
        "platform": platform.system(),
        "python": platform.python_version(),
        "inventory_counts": expected,
        "configuration": profile_summary(profile),
        "write_rejected": True,
        "lan_discovery_performed": False,
        "training_stack_imported": training_imported,
        "physical_commands_sent": 0,
        "device_control_acceptance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="Check synthetic loopback reads and write rejection")
    connect = commands.add_parser(
        "connect", help="Discover LAN gateways and save private read-only inventory"
    )
    connect.add_argument("--root", type=Path, required=True)
    connect.add_argument("--interface", help="Local computer IPv4 address, not gateway address")
    inspect = commands.add_parser("inspect", help="Read inventory counts from an already known gateway")
    inspect.add_argument("--host", required=True)
    inspect.add_argument("--port", type=int, default=65443)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "src"
    required = (
        "contracts",
        "data",
        "home",
        "home_mock",
        "home_profile",
        "home_connect",
        "yeelight",
        "yeelight_discovery",
        "__init__",
    )
    if any(not (source / "ldm" / (name + ".py")).is_file() for name in required):
        print(json.dumps({"error_code": "incomplete_lan_handoff_source"}))
        return 1
    sys.path.insert(0, str(source))
    from ldm.contracts import DomainError

    try:
        if args.command == "self-test":
            result = self_test()
        elif args.command == "connect":
            from ldm.home_connect import home_connect

            result = home_connect(args.root, args.interface)
        else:
            from ldm.yeelight import YeelightReader

            result = YeelightReader(args.host, args.port).summary()
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 2 if args.command == "connect" and result["status"] != "read_only_connected" else 0
    except DomainError as error:
        print(json.dumps({"error_code": error.code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
