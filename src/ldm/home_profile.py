"""Private, read-only configuration snapshot from the documented Pro LAN inventory.

This is an input contract, not a controller, action catalogue or training-label
generator. A listed scene does not expose its actions. Unknowns stay unknown.
"""

import re
from datetime import datetime

from .contracts import DomainError

# Device type meanings: owner-supplied Pro LAN protocol v2.22, pp. 6, 19-25.
KINDS = {
    1: "switchable_light",
    2: "dimmable_light",
    3: "color_temperature_light",
    4: "color_light",
    6: "curtain_motor",
    7: "dual_relay",
    10: "multi_channel_air_conditioning_gateway",
    13: "multi_channel_switch",
    14: "focusable_color_temperature_light",
    15: "air_conditioning_controller",
    128: "scene_panel",
    129: "motion_or_presence_sensor",
    130: "contact_sensor",
    132: "rotary_switch",
    134: "motion_and_light_level_sensor",
    135: "brightness_sensor",
    136: "temperature_humidity_sensor",
}
COLLECTION_TYPES = {"nodes": 2, "rooms": 1, "groups": 4, "scenes": 6}
# Bounds are reported as a band, never converted to an exact lux reading.
# Endpoint inclusivity is not established for the middle bands by the PDF.
LIGHT_LEVEL_BANDS = {
    1: (None, 23),
    2: (23, 50),
    3: (50, 150),
    4: (150, 250),
    5: (250, 350),
    6: (350, 450),
    7: (450, 600),
    8: (600, None),
}


def _bool(value):
    return value if type(value) is bool else None


def _integer(value, low, high):
    return value if type(value) is int and low <= value <= high else None


def _properties(product_type: int | None, params: dict) -> tuple[dict, list[str]]:
    """Normalize a deliberately limited, documented state subset; never guess."""
    state, mapped = {}, set()

    def field(wire, name, transform):
        mapped.add(wire)
        state[name] = transform(params.get(wire))

    if product_type in {1, 2, 3, 4, 14}:
        field("p", "power", _bool)
        if product_type != 1:
            field("l", "brightness_percent", lambda x: _integer(x, 1, 100))
        if product_type in {3, 4, 14}:
            field("ct", "color_temperature_kelvin", lambda x: _integer(x, 2700, 6500))
        if product_type == 4:
            field("c", "color_rgb_integer", lambda x: _integer(x, 0, 16777215))
    elif product_type == 6:
        field("cp", "position_percent", lambda x: _integer(x, 0, 100))
        field("tp", "target_position_percent", lambda x: _integer(x, 0, 100))
        # The installation's opening direction is not defined by this snapshot.
        state["fully_open_at_percent"] = None
    elif product_type in {129, 134}:
        field("mv", "motion_detected", lambda x: None if _integer(x, 0, 1) is None else bool(x))
        state["occupancy"] = None
        if product_type == 134:
            field("level", "light_level", lambda x: _integer(x, 1, 8))
            level = state["light_level"]
            state["illuminance_lux"] = None
            state["illuminance_band_lux"] = (
                {
                    "lower": LIGHT_LEVEL_BANDS[level][0],
                    "upper": LIGHT_LEVEL_BANDS[level][1],
                    "endpoint_inclusivity": "unverified",
                }
                if level
                else None
            )
    elif product_type == 130:
        field("dc", "contact_closed", lambda x: None if _integer(x, 0, 1) is None else bool(x))
    elif product_type == 136:
        field("h", "humidity_percent", lambda x: _integer(x, 0, 100))
        field("t", "temperature_celsius", lambda x: None if _integer(x, -1000, 5000) is None else x / 100)
    return state, sorted(set(params) - mapped)


def build_home_profile(inventory: dict, connection_id: str, observed_at: str) -> dict:
    """All returned rows are private. Only profile_summary may reach the console.

    IDs are namespaced by gateway AND node type. Room/group membership is not
    inferred from names, collection order or a similarly numbered device.
    """
    try:
        if not re.fullmatch(r"[0-9a-f]{32}", connection_id):
            raise ValueError
        if datetime.fromisoformat(observed_at).tzinfo is None or not isinstance(inventory, dict):
            raise ValueError
        collections, seen = {}, set()
        for collection, node_type in COLLECTION_TYPES.items():
            rows = inventory.get(collection)
            if rows is None:
                collections[collection] = None
                continue
            if not isinstance(rows, list):
                raise ValueError
            normalized = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError
                identifier = row.get("id")
                if _integer(identifier, 1, 2**64 - 1) is None:
                    raise ValueError
                if "nt" in row and (type(row["nt"]) is not int or row["nt"] != node_type):
                    raise ValueError
                ref = f"{connection_id}/{node_type}/{identifier}"
                if ref in seen:
                    raise ValueError
                seen.add(ref)
                params = row.get("params", {})
                if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
                    raise ValueError
                product_type = row.get("pt") if type(row.get("pt")) is int else None
                state, unmapped = _properties(product_type, params)
                item = {
                    "ref": ref,
                    "name": row.get("n") if isinstance(row.get("n"), str) else None,
                    "node_type": node_type,
                    "product_type": product_type,
                    "kind": KINDS.get(product_type, "unknown")
                    if collection in {"nodes", "groups"}
                    else {"rooms": "room", "scenes": "scene"}[collection],
                    "online": _bool(row.get("o")),
                    "state": state,
                    "unmapped_property_names": unmapped,
                    "state_fresh": None,
                    "device_reported_at": None,
                }
                if collection == "nodes":
                    item.update(room_ref=None, group_refs=None)
                elif collection in {"rooms", "groups"}:
                    item["member_refs"] = None
                else:
                    value = params.get("state")
                    item["state"] = {
                        "scene_status": value if value in ("active", "inactive", "unknown") else None
                    }
                    item["unmapped_property_names"] = sorted(set(params) - {"state"})
                    item.update(actions=None, action_detail_status="not_defined_in_lan_v2_22")
                normalized.append(item)
            collections[collection] = normalized
    except (KeyError, TypeError, ValueError, OverflowError):
        raise DomainError("home_profile_invalid_inventory") from None
    return {
        "schema_version": 1,
        "source": "yeelight_pro_lan_v2_22_inventory",
        "connection_id": connection_id,
        "observed_at": observed_at,
        "observation_semantics": "sequential_gateway_reads_not_atomic_device_timestamps",
        "collections": collections,
        "automations": {"status": "not_defined_in_lan_v2_22", "rules": None},
        "permission_granted": False,
        "training_ready": False,
        "device_control_acceptance": False,
    }


def profile_summary(profile: dict) -> dict:
    """Aggregate allowlist; never return names, values, wire IDs or error text."""
    collections = profile["collections"]
    devices = (collections["nodes"] or []) + (collections["groups"] or [])
    return {
        "status": "partial_configuration",
        "inventory_counts": {k: None if v is None else len(v) for k, v in collections.items()},
        "unknown_device_types": sum(row["kind"] == "unknown" for row in devices),
        "entities_with_unmapped_properties": sum(bool(row["unmapped_property_names"]) for row in devices),
        "scene_actions_available": False,
        "automation_rules_available": False,
        "membership_available": False,
        "device_freshness_verified": False,
        "training_ready": False,
        "control_permission_granted": False,
    }
