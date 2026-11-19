import json
import socketserver
import threading
from contextlib import contextmanager

import pytest

from ldm.contracts import DomainError, HomeRequest
from ldm.home_mock import MockGateway, home_mock
from ldm.yeelight import YeelightReader, exchange


@contextmanager
def scripted_gateway(reply):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            for fragment in reply(request):
                self.wfile.write(fragment)
                self.wfile.flush()

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield YeelightReader(*server.server_address)
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_wire_replay_requires_approval_and_reads_back_state(tmp_path):
    result = home_mock(tmp_path)
    assert result["inventory_counts"] == {"nodes": 2, "rooms": 1, "groups": 1, "scenes": 1}
    assert [row["result"]["status"] for row in result["blocked_or_deferred"]] == [
        "confirmation_required",
        "denied",
        "clarify",
        "manual_fallback",
        "denied",
    ]
    assert result["confirmed_reading"]["state"] == {"p": True, "l": 70, "ct": 4000}
    assert result["mock_commands_sent"] == 1
    assert result["physical_commands_sent"] == 0
    assert result["offline"]["reason"] == "mock_device_offline"
    assert result["physical_device_acceptance"] is False
    assert json.loads((tmp_path / "home-mock-receipt.json").read_text()) == result


def test_live_reader_rejects_writes_before_opening_a_connection(monkeypatch):
    def forbidden_connection(*_args, **_kwargs):
        pytest.fail("a write reached the read-only transport")

    monkeypatch.setattr("ldm.yeelight.exchange", forbidden_connection)
    with pytest.raises(DomainError, match="gateway_read_only"):
        YeelightReader("127.0.0.1").query("gateway_set.prop")


def test_low_confidence_wrong_space_and_offline_leave_mock_state_unchanged():
    with MockGateway() as gateway:
        before = gateway.reader().query("gateway_get.node", 101)
        confirmed = HomeRequest(intent="reading", human_confirmed=True)
        assert gateway.reading(confirmed, "owner", [0.34, 0.33, 0.33])["status"] == "clarify"
        assert (
            gateway.reading(
                HomeRequest(intent="reading", space="guest_room", human_confirmed=True),
                "owner",
                [0.95, 0.03, 0.02],
            )["status"]
            == "denied"
        )
        gateway.set_online(False)
        assert gateway.reading(confirmed, "owner", [0.95, 0.03, 0.02])["status"] == "manual_fallback"
        assert gateway.reader().query("gateway_get.node", 101)[0]["params"] == before[0]["params"]
        assert gateway.mock_commands_sent == 0


def test_invalid_mock_write_is_atomic():
    with MockGateway() as gateway:
        initial = gateway.reader().query("gateway_get.node", 101)
        with pytest.raises(DomainError, match="gateway_request_rejected"):
            exchange(
                *gateway.address,
                {
                    "id": 7,
                    "method": "gateway_set.prop",
                    "nodes": [{"id": 101, "nt": 2, "set": {"p": True, "l": 0}}],
                },
            )
        assert gateway.reader().query("gateway_get.node", 101) == initial
        assert gateway.mock_commands_sent == 0


def test_client_correlates_ids_and_reassembles_fragmented_frames():
    def reply(request):
        notification = b'{"method":"gateway_post.event","id":999,"nodes":[]}\r\n'
        response = json.dumps({"id": request["id"], "rooms": [{"id": 8, "n": "Synthetic"}]}).encode()
        return [notification + response[:5], response[5:17], response[17:] + b"\r\n"]

    with scripted_gateway(reply) as reader:
        assert reader.query("gateway_get.room") == [{"id": 8, "n": "Synthetic"}]


@pytest.mark.parametrize(
    "response,code",
    [
        (b"PRIVATE_RAW_ERROR\r\n", "gateway_protocol_or_connection_error"),
        (b"[]\r\n", "gateway_invalid_response"),
        (b'{"id":1,"rooms":"PRIVATE_RAW_RECORD"}\r\n', "gateway_invalid_inventory"),
        (b"", "gateway_connection_closed"),
    ],
)
def test_malformed_and_closed_gateway_responses_fail_without_echo(response, code):
    with scripted_gateway(lambda _request: [response]) as reader:
        with pytest.raises(DomainError) as error:
            reader.query("gateway_get.room")
    assert error.value.code == code
    assert "PRIVATE" not in str(error.value)


def test_client_limits_total_response_bytes(monkeypatch):
    monkeypatch.setattr("ldm.yeelight.MAX_RESPONSE_BYTES", 64)
    with scripted_gateway(lambda _request: [b"x" * 65]) as reader:
        with pytest.raises(DomainError, match="gateway_response_too_large"):
            reader.query("gateway_get.room")


def test_gateway_silence_is_bounded_by_the_read_budget():
    accepted = threading.Event()
    release = threading.Event()

    def no_reply(_request):
        accepted.set()
        release.wait(5)
        return []

    with scripted_gateway(no_reply) as reader:
        reader.timeout = 1.0
        try:
            with pytest.raises(DomainError, match="gateway_timeout"):
                reader.query("gateway_get.room")
        finally:
            release.set()
    assert accepted.is_set()


@pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "0.0.0.0", "224.0.0.251"])
def test_gateway_must_be_an_explicit_local_address(host):
    with pytest.raises(DomainError, match="gateway_private_ipv4_required"):
        YeelightReader(host)
