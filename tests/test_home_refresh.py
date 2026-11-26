from copy import deepcopy

import pytest

from ldm.home_refresh import MockObservation, review_lamp


def lamp_request():
    return {
        "home_profile": {"synthetic": True},
        "user_request": "开灯",
        "trigger": "user_request",
        "devices": [{"id": "mock-light-101", "kind": "light", "online": True, "properties": {"p": False}}],
    }


def observation():
    return MockObservation(
        [{"id": 101, "nt": 2, "pt": 3, "o": True, "params": {"p": True, "l": 30, "ct": 3000}}], True
    )


def test_unknown_reading_overrides_first_write_and_keeps_request():
    initial = lamp_request()
    seen = []

    def choose(s):
        seen.append(deepcopy(s))
        return "keep" if s["devices"][0]["properties"]["p"] else "turn_on"

    result = review_lamp(initial, choose, observation)
    assert result["refresh_reason"] == "freshness_guard" and result["reads"] == 1
    assert result["suggestion"] == "keep" and result["choices"] == ["turn_on", "keep"]
    assert initial == seen[0] == lamp_request()
    expected = deepcopy(initial)
    expected["devices"][0].update(state_fresh=True, properties={"p": True, "l": 30, "ct": 3000})
    assert seen[1] == expected


@pytest.mark.parametrize(
    "problem", ["cached", "offline", "wrong_target", "invalid_params", "timeout", "refresh_again"]
)
def test_uncertain_observation_or_second_refresh_stops(problem):
    reads = []

    def observe():
        reads.append(True)
        r = observation()
        if problem == "timeout":
            raise TimeoutError
        if problem == "offline":
            r.nodes[0]["o"] = False
        if problem == "wrong_target":
            r.nodes[0]["id"] = 102
        if problem == "invalid_params":
            r.nodes[0]["params"]["p"] = 1
        return MockObservation(r.nodes, problem != "cached")

    result = review_lamp(lamp_request(), lambda _: "refresh_state", observe)
    assert result["status"] == "manual_review" and result["suggestion"] is None
    assert reads == [True]
    assert len(result["choices"]) == (2 if problem == "refresh_again" else 1)
