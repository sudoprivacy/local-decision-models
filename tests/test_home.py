import pytest

from ldm.contracts import DomainError, HomeRequest
from ldm.home import policy


@pytest.mark.parametrize(
    "payload,role,expected",
    [
        ({"intent": "reading"}, "owner", "confirmation_required"),
        ({"intent": "reading", "human_confirmed": True}, "owner", "simulated"),
        ({"intent": "reading", "human_confirmed": True}, "guest", "denied"),
        ({"intent": "ambiguous"}, "owner", "clarify"),
        ({"intent": "reading", "model_online": False}, "owner", "manual_fallback"),
        ({"intent": "open_entry", "human_confirmed": True}, "owner", "denied"),
        (
            {"intent": "reading", "space": "guest_room", "human_confirmed": True},
            "guest",
            "confirmation_required",
        ),
    ],
)
def test_model_confidence_never_grants_authority(payload, role, expected):
    result = policy(HomeRequest(**payload), role, [0.95, 0.03, 0.02])
    assert result["status"] == expected
    assert result["physical_commands_sent"] == 0
    assert result["manual_path_available"]


def test_uncertain_or_invalid_distribution_cannot_execute():
    request = HomeRequest(intent="reading", human_confirmed=True)
    assert policy(request, "owner", [0.34, 0.33, 0.33])["status"] == "clarify"
    with pytest.raises(DomainError, match="invalid_model_decision"):
        policy(request, "owner", [float("nan"), 0, 1])


def test_http_identity_cannot_be_spoofed_by_body(client, auth):
    response = client.post(
        "/v1/home/preview", headers=auth["guest"], json={"intent": "reading", "role": "owner"}
    )
    assert response.status_code == 422
    result = client.post(
        "/v1/home/preview", headers=auth["guest"], json={"intent": "reading", "human_confirmed": True}
    )
    assert result.json()["status"] == "denied"
