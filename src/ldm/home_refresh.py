"""A bounded, read-only decision loop for the single simulated study lamp."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from .contracts import DomainError


@dataclass(frozen=True)
class MockObservation:
    nodes: list[dict]
    # True only for an authoritative observation from the owned simulator.
    verified_current: bool


def review_lamp(state: dict, choose: Callable[[dict], str], observe: Callable[[], MockObservation]) -> dict:
    current = deepcopy(state)
    devices = current.get("devices", [])
    if (
        current.get("home_profile", {}).get("synthetic") is not True
        or len(devices) != 1
        or devices[0].get("id") != "mock-light-101"
        or devices[0].get("kind") != "light"
        or not isinstance(current.get("user_request"), str)
        or not current["user_request"].strip()
    ):
        raise DomainError("single_mock_lamp_request_required")
    lamp = devices[0]
    trace = {"choices": [], "reads": 0, "refresh_reason": None, "suggestion": None}
    if lamp.get("online") is not True:
        return {**trace, "status": "manual_review", "reason": "availability_unverified"}
    action = choose(deepcopy(current))
    trace["choices"].append(action)
    if lamp.get("state_fresh") is True and action != "refresh_state":
        return {**trace, "status": "review_suggestion", "suggestion": action}

    trace["refresh_reason"] = "model" if action == "refresh_state" else "freshness_guard"
    trace["reads"] = 1
    try:
        observation = observe()
    except (DomainError, OSError):
        return {**trace, "status": "manual_review", "reason": "read_failed"}
    if len(observation.nodes) != 1:
        return {**trace, "status": "manual_review", "reason": "target_unverified"}
    node = observation.nodes[0]
    if (node.get("id"), node.get("nt"), node.get("pt")) != (101, 2, 3):
        return {**trace, "status": "manual_review", "reason": "target_unverified"}
    if node.get("o") is not True or observation.verified_current is not True:
        return {**trace, "status": "manual_review", "reason": "observation_unverified"}
    props = node.get("params")
    if (
        not isinstance(props, dict)
        or type(props.get("p")) is not bool
        or type(props.get("l")) is not int
        or not 1 <= props["l"] <= 100
        or type(props.get("ct")) is not int
        or not 2700 <= props["ct"] <= 6500
    ):
        return {**trace, "status": "manual_review", "reason": "incomplete_lamp_state"}
    lamp["properties"] = {key: props[key] for key in ("p", "l", "ct")}
    lamp["state_fresh"] = True
    action = choose(current)
    trace["choices"].append(action)
    if action == "refresh_state":
        return {**trace, "status": "manual_review", "reason": "still_requests_refresh"}
    return {**trace, "status": "review_suggestion", "suggestion": action}
