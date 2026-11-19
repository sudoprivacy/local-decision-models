"""Discover and inspect home gateways locally; connection conveys no authority."""

import json
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .contracts import DomainError
from .data import atomic_json
from .home_profile import build_home_profile, profile_summary
from .yeelight import YeelightReader
from .yeelight_discovery import Gateway, discover_gateways


def inspect_gateway(gateway: Gateway) -> dict:
    try:
        reader = YeelightReader(gateway.host, timeout=1.0)
        topology = reader.query("gateway_get.topology")
        inventory = reader.inventory()
        return {
            "status": "read_only_connected",
            "topology_count": len(topology),
            "inventory_counts": {key: len(rows) for key, rows in inventory.items()},
            "inventory": inventory,
            "topology": topology,
        }
    except DomainError as error:
        return {"status": "local_api_unavailable", "error_code": error.code}


def read_registry(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "gateways": {}}
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry["version"] != 1 or not isinstance(registry["gateways"], dict):
            raise ValueError
        connection_ids = set()
        for identity, record in registry["gateways"].items():
            if (
                not isinstance(identity, str)
                or not isinstance(record, dict)
                or not re.fullmatch(r"[0-9a-f]{32}", record.get("connection_id", ""))
            ):
                raise ValueError
            if record["connection_id"] in connection_ids:
                raise ValueError
            connection_ids.add(record["connection_id"])
        return registry
    except (OSError, ValueError, KeyError, TypeError):
        raise DomainError("gateway_private_registry_invalid") from None


def connect_gateways(root: Path, interface: str | None) -> dict:
    registry_path = root / "gateways.private.json"
    registry = read_registry(registry_path)
    for record in registry["gateways"].values():
        record["observed_this_run"] = False
    summary = {
        "mode": "local_gateway_onboarding",
        "route": "direct_lan_tcp",
        "app_used": False,
        "cloud_used": False,
        "mcp_used": False,
        "onboarding_requirements": {
            "vendor_qr_authorization": "not_required_for_direct_lan",
            "local_control": "must_already_be_accessible",
            "automatic_local_control_enable_implemented": False,
        },
        "new_home_zero_touch_verified": False,
        "physical_commands_sent": 0,
        "control_permission_granted": False,
        "device_control_acceptance": False,
        "gateways": [],
    }
    try:
        discovery = discover_gateways(interface=interface)
    except DomainError as error:
        summary.update(
            status="discovery_failed",
            error_code=error.code,
            next_step="check_local_network_interfaces",
        )
        atomic_json(registry_path, registry)
        atomic_json(root / "home-connection-receipt.json", summary)
        return summary
    summary.update(
        interfaces_checked=discovery.interfaces_checked,
        discovery_complete=discovery.interface_failures == 0,
        interface_failures=discovery.interface_failures,
        gateways_discovered=len(discovery.gateways),
    )
    identities = Counter(item.identity for item in discovery.gateways)
    hosts = Counter(item.host for item in discovery.gateways)

    def inspect(candidate):
        if identities[candidate.identity] > 1 or hosts[candidate.host] > 1:
            return {"status": "identity_conflict", "error_code": "gateway_discovery_identity_conflict"}
        return inspect_gateway(candidate)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(inspect, discovery.gateways))
    for candidate, result in zip(discovery.gateways, results, strict=True):
        record = registry["gateways"].setdefault(candidate.identity, {"connection_id": uuid.uuid4().hex})
        record.update(
            **asdict(candidate),
            observed_this_run=True,
            last_seen_at=datetime.now(UTC).isoformat(),
            status=result["status"],
        )
        profile_path = root / "gateways" / record["connection_id"] / "profile.private.json"
        if result["status"] == "read_only_connected":
            atomic_json(root / "gateways" / record["connection_id"] / "inventory.private.json", result)
            try:
                profile = build_home_profile(
                    result["inventory"], record["connection_id"], record["last_seen_at"]
                )
                atomic_json(profile_path, profile)
                result["configuration"] = profile_summary(profile)
            except DomainError as error:
                result["configuration"] = {"status": "unavailable", "error_code": error.code}
        else:
            result["configuration"] = {"status": "unavailable"}
        if result["configuration"]["status"] == "unavailable":
            # A failed refresh must not leave an older profile looking current.
            atomic_json(
                profile_path,
                {
                    "schema_version": 1,
                    "connection_id": record["connection_id"],
                    "observed_at": record["last_seen_at"],
                    "status": "unavailable",
                    "training_ready": False,
                    "permission_granted": False,
                },
            )
        # No IP, MAC, device identities, room names, inventory rows or raw errors
        # enter the console summary or research API. Cached IPs are never probed.
        summary["gateways"].append(
            {
                "connection_id": record["connection_id"],
                **{key: value for key, value in result.items() if key not in {"inventory", "topology"}},
            }
        )
    connected = sum(row["status"] == "read_only_connected" for row in summary["gateways"])
    if not discovery.gateways:
        status = "no_gateway_discovered" if summary["discovery_complete"] else "discovery_incomplete"
    elif connected == len(discovery.gateways) and summary["discovery_complete"]:
        status = "read_only_connected"
    else:
        status = "partially_connected" if connected else "local_access_unavailable"
    summary.update(status=status, gateways_connected=connected)
    summary["existing_local_api_connection_verified"] = connected > 0
    summary["next_step"] = {
        "read_only_connected": "local_inventory_ready",
        "no_gateway_discovered": "check_same_lan_broadcast_and_local_control",
        "discovery_incomplete": "check_local_network_interfaces",
        "partially_connected": "review_per_gateway_local_access",
        "local_access_unavailable": "check_local_control_network_and_firmware",
    }[status]
    if any(row["status"] == "identity_conflict" for row in summary["gateways"]):
        summary["next_step"] = "resolve_gateway_identity_conflict"
    atomic_json(registry_path, registry)
    atomic_json(root / "home-connection-receipt.json", summary)
    return summary


def home_connect(root: Path, interface: str | None = None) -> dict:
    root = root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise DomainError("runtime_must_be_outside_checkout")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise DomainError("gateway_connection_storage_error") from None
    lock = root / "home-connect.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise DomainError("gateway_connection_profile_busy") from None
    except OSError:
        raise DomainError("gateway_connection_storage_error") from None
    try:
        with stream:
            stream.write("Local read-only discovery in progress.\n")
            return connect_gateways(root, interface)
    except OSError:
        raise DomainError("gateway_connection_storage_error") from None
    finally:
        lock.unlink()
