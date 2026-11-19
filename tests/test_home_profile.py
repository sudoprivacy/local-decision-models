import copy
import json

import pytest

from ldm import home_connect as onboarding
from ldm.contracts import DomainError
from ldm.home_mock import MockGateway
from ldm.home_profile import build_home_profile, profile_summary
from ldm.yeelight_discovery import Discovery, Gateway

CONNECTION = "a" * 32
OBSERVED = "2026-11-14T16:00:00+00:00"


def inventory():
    # Entirely invented fixtures; no screenshot identities, configuration or labels.
    return {
        "nodes": [
            {
                "id": 11,
                "nt": 2,
                "n": "Fixture lamp",
                "pt": 3,
                "o": True,
                "params": {"p": False, "l": 42, "ct": 3500},
            },
            {"id": 12, "nt": 2, "n": "Fixture sensor", "pt": 134, "o": True, "params": {"mv": 0, "level": 4}},
            {"id": 13, "nt": 2, "n": "Fixture curtain", "pt": 6, "params": {"cp": 100}},
            {
                "id": 14,
                "nt": 2,
                "n": "Fixture HVAC",
                "pt": 10,
                "o": False,
                "params": {"1-acp": False, "2-actt": 24},
            },
            {"id": 15, "nt": 2, "n": "Fixture new type", "pt": 999, "params": {"p": True, "lux": 123}},
            {"id": 2**63 + 17, "nt": 2, "n": "Fixture climate", "pt": 136, "params": {"t": 2134, "h": 53}},
        ],
        "rooms": [{"id": 11, "n": "Fixture lamp"}],
        "groups": [{"id": 11, "nt": 4, "n": "Fixture lamp", "pt": 3, "params": {}}],
        "scenes": [{"id": 11, "n": "Entire fixture off", "params": {"state": "active"}}],
    }


def profile(data=None, connection=CONNECTION):
    return build_home_profile(data if data is not None else inventory(), connection, OBSERVED)


def test_mixed_inventory_over_socket_preserves_unknowns_and_does_not_write():
    with MockGateway() as gateway:
        for collection, rows in inventory().items():
            setattr(gateway, collection, rows)
        result = profile(gateway.reader().inventory())
        nodes = result["collections"]["nodes"]
        assert nodes[0]["online"] is True and nodes[0]["state"]["power"] is False
        assert nodes[1]["state"]["motion_detected"] is False
        assert nodes[1]["state"]["occupancy"] is None
        assert nodes[1]["state"]["illuminance_lux"] is None
        assert nodes[1]["state"]["illuminance_band_lux"]["lower"] == 150
        assert nodes[1]["state"]["illuminance_band_lux"]["upper"] == 250
        assert nodes[2]["online"] is None
        assert nodes[2]["state"]["fully_open_at_percent"] is None
        assert nodes[3]["kind"] == "multi_channel_air_conditioning_gateway"
        assert nodes[3]["state"] == {}
        assert nodes[3]["unmapped_property_names"] == ["1-acp", "2-actt"]
        assert nodes[4]["kind"] == "unknown" and nodes[4]["state"] == {}
        assert nodes[5]["state"] == {"temperature_celsius": 21.34, "humidity_percent": 53}
        assert nodes[5]["ref"].endswith(str(2**63 + 17))
        assert gateway.mock_commands_sent == 0
    assert all(row["state_fresh"] is None for rows in result["collections"].values() for row in rows)
    assert result["training_ready"] is False


def test_scene_names_and_active_status_do_not_supply_actions_or_members():
    result = profile()
    collections = result["collections"]
    scene = collections["scenes"][0]
    assert scene["state"]["scene_status"] == "active"
    assert scene["actions"] is None
    assert collections["nodes"][0]["room_ref"] is None
    assert collections["nodes"][0]["group_refs"] is None
    assert collections["rooms"][0]["member_refs"] is None
    assert collections["groups"][0]["member_refs"] is None
    assert result["automations"]["rules"] is None
    assert not result["permission_granted"]
    # Unknown vendor extension data cannot silently become a supported action.
    data = inventory()
    data["scenes"][0]["actions"] = [{"target": "entire_home", "power": False}]
    assert profile(data)["collections"]["scenes"][0]["actions"] is None


def test_identity_is_gateway_and_type_scoped_not_name_or_position():
    first = profile()
    second = profile(connection="b" * 32)
    refs = [r["ref"] for p in (first, second) for rows in p["collections"].values() for r in rows]
    assert len(refs) == len(set(refs))
    assert first["collections"]["nodes"][0]["ref"] != first["collections"]["rooms"][0]["ref"]


def test_missing_collections_are_distinct_from_observed_empty_collections():
    missing = profile_summary(profile({}))
    empty = profile_summary(profile({k: [] for k in inventory()}))
    assert missing["inventory_counts"] == {k: None for k in inventory()}
    assert empty["inventory_counts"] == {k: 0 for k in inventory()}
    for result in (missing, empty):
        assert result["status"] == "partial_configuration"
        assert not result["automation_rules_available"]
        assert not result["scene_actions_available"]
        assert not result["training_ready"]


@pytest.mark.parametrize("mv,level", [(False, True), ("0", "4"), (2, 9), (None, None)])
def test_invalid_or_missing_sensor_values_do_not_become_measurements(mv, level):
    data = inventory()
    data["nodes"][1]["params"] = {"mv": mv, "level": level}
    state = profile(data)["collections"]["nodes"][1]["state"]
    assert state["motion_detected"] is None
    assert state["illuminance_band_lux"] is None
    assert state["occupancy"] is None


@pytest.mark.parametrize("properties", [{}, {"p": "false", "l": 0, "ct": 9000}])
def test_unknown_lamp_state_does_not_get_default_settings(properties):
    data = inventory()
    data["nodes"][0]["params"] = properties
    assert all(v is None for v in profile(data)["collections"]["nodes"][0]["state"].values())


@pytest.mark.parametrize("change", ["duplicate", "bad_nt", "bad_id", "bad_params", "bad_collection"])
def test_malformed_identity_or_structure_is_rejected_without_private_echo(change):
    data = inventory()
    if change == "duplicate":
        data["nodes"].append(copy.deepcopy(data["nodes"][0]))
    elif change == "bad_nt":
        data["nodes"][0]["nt"] = 4
    elif change == "bad_id":
        data["nodes"][0]["id"] = "PRIVATE_INVALID_ID"
    elif change == "bad_params":
        data["nodes"][0]["params"] = "PRIVATE_INVALID_PARAMS"
    else:
        data["nodes"] = {"PRIVATE": "not an array"}
    with pytest.raises(DomainError, match="^home_profile_invalid_inventory$"):
        profile(data)


def test_summary_is_aggregate_and_input_unchanged():
    data = inventory()
    before = copy.deepcopy(data)
    result = profile_summary(profile(data))
    assert data == before
    assert result["unknown_device_types"] == 1
    assert result["entities_with_unmapped_properties"] == 2
    public = json.dumps(result)
    for forbidden in ("Fixture", CONNECTION, "3500", "2134", "1-acp", "Entire", str(2**63 + 17)):
        assert forbidden not in public


def test_onboarding_saves_private_profile_and_only_returns_aggregate(tmp_path, monkeypatch):
    candidate = Gateway("192.168.50.20", "00:00:00:00:00:11", "fixture-did", "1")
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery([candidate], 1))
    monkeypatch.setattr(
        onboarding,
        "inspect_gateway",
        lambda _candidate: {
            "status": "read_only_connected",
            "inventory": inventory(),
            "topology": [],
            "topology_count": 0,
            "inventory_counts": {k: len(v) for k, v in inventory().items()},
        },
    )
    receipt = onboarding.home_connect(tmp_path)
    connection = receipt["gateways"][0]["connection_id"]
    stored = json.loads((tmp_path / "gateways" / connection / "profile.private.json").read_text())
    assert stored["collections"]["nodes"][0]["name"] == "Fixture lamp"
    assert receipt["gateways"][0]["configuration"]["status"] == "partial_configuration"
    assert "Fixture" not in json.dumps(receipt)
    assert receipt["physical_commands_sent"] == 0


def test_profile_failure_preserves_raw_read_success(tmp_path, monkeypatch):
    candidate = Gateway("192.168.50.20", "00:00:00:00:00:11", "fixture-did", "1")
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery([candidate], 1))
    monkeypatch.setattr(
        onboarding,
        "inspect_gateway",
        lambda _candidate: {
            "status": "read_only_connected",
            "inventory": {"nodes": [{"id": "PRIVATE_BAD_ID"}]},
            "topology": [],
            "inventory_counts": {"nodes": 1},
        },
    )
    receipt = onboarding.home_connect(tmp_path)
    assert receipt["status"] == "read_only_connected"
    assert receipt["gateways"][0]["configuration"] == {
        "status": "unavailable",
        "error_code": "home_profile_invalid_inventory",
    }
    assert "PRIVATE_BAD_ID" not in json.dumps(receipt)


@pytest.mark.parametrize("failure", ["local_api_unavailable", "invalid_inventory"])
def test_failed_refresh_invalidates_previous_configuration(tmp_path, monkeypatch, failure):
    candidate = Gateway("192.168.50.20", "00:00:00:00:00:11", "fixture-did", "1")
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery([candidate], 1))
    result = {"status": "read_only_connected", "inventory": inventory(), "topology": []}
    monkeypatch.setattr(onboarding, "inspect_gateway", lambda _candidate: copy.deepcopy(result))
    first = onboarding.home_connect(tmp_path)
    connection = first["gateways"][0]["connection_id"]
    path = tmp_path / "gateways" / connection / "profile.private.json"
    assert "collections" in json.loads(path.read_text())
    if failure == "local_api_unavailable":
        result = {"status": failure, "error_code": "gateway_timeout"}
    else:
        result["inventory"] = {"nodes": [{"id": False}]}
    current = onboarding.home_connect(tmp_path)
    assert current["gateways"][0]["configuration"]["status"] == "unavailable"
    assert json.loads(path.read_text())["status"] == "unavailable"
    assert "collections" not in json.loads(path.read_text())
