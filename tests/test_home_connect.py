import ipaddress
import json
import socket
import socketserver
import threading
from types import SimpleNamespace

import pytest

import ldm.home_connect as onboarding
import ldm.yeelight_discovery as discovery
from ldm.contracts import DomainError
from ldm.home_mock import MockGateway
from ldm.yeelight import YeelightReader
from ldm.yeelight_discovery import Discovery, Gateway, LanInterface

SYNTHETIC_MAC = "02:00:00:00:00:01"
LAN = LanInterface("192.168.50.10", ipaddress.IPv4Network("192.168.50.0/24"))


def advertisement(host="192.168.50.20"):
    return f"pid:14337\r\nmac:{SYNTHETIC_MAC}\r\ndid:synthetic-1\r\nip:{host}\r\n".encode()


def synthetic_gateway(host="192.168.50.20", did="synthetic-1"):
    return Gateway(host, SYNTHETIC_MAC, did, "14337")


def test_onboarding_reads_mock_over_tcp_and_keeps_inventory_private(tmp_path, monkeypatch):
    with MockGateway() as gateway:
        monkeypatch.setattr(
            onboarding, "discover_gateways", lambda **_kwargs: Discovery([synthetic_gateway()], 1)
        )

        def local_reader(host, timeout):
            assert host == "192.168.50.20"
            return YeelightReader(*gateway.address, timeout=timeout)

        monkeypatch.setattr(onboarding, "YeelightReader", local_reader)
        result = onboarding.home_connect(tmp_path)
        assert gateway.mock_commands_sent == 0
    assert result["status"] == "read_only_connected"
    assert result["gateways"][0]["inventory_counts"] == {"nodes": 2, "rooms": 1, "groups": 1, "scenes": 1}
    assert result["physical_commands_sent"] == 0
    assert result["control_permission_granted"] is False
    assert not result["app_used"] and not result["cloud_used"] and not result["mcp_used"]
    assert result["onboarding_requirements"]["vendor_qr_authorization"] == "not_required_for_direct_lan"
    assert result["onboarding_requirements"]["automatic_local_control_enable_implemented"] is False
    assert result["existing_local_api_connection_verified"] is True
    assert result["new_home_zero_touch_verified"] is False
    public = json.dumps(result)
    for private in ("192.168.50.20", SYNTHETIC_MAC, "synthetic-1", "Synthetic study light"):
        assert private not in public
    connection_id = result["gateways"][0]["connection_id"]
    stored = json.loads((tmp_path / "gateways" / connection_id / "inventory.private.json").read_text())
    assert stored["inventory"]["nodes"][0]["n"] == "Synthetic study light"
    assert json.loads((tmp_path / "home-connection-receipt.json").read_text()) == result
    assert not (tmp_path / "home-connect.lock").exists()


def test_dhcp_change_keeps_identity_but_cached_addresses_are_never_probed(tmp_path, monkeypatch):
    current, probed = [], []
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery(list(current), 1))

    def inspect(candidate):
        probed.append(candidate.host)
        return {"status": "read_only_connected", "inventory": {}, "topology": [], "inventory_counts": {}}

    monkeypatch.setattr(onboarding, "inspect_gateway", inspect)
    current[:] = [synthetic_gateway()]
    first_id = onboarding.home_connect(tmp_path)["gateways"][0]["connection_id"]
    current[:] = [synthetic_gateway("192.168.50.30")]
    assert onboarding.home_connect(tmp_path)["gateways"][0]["connection_id"] == first_id
    current[:] = [synthetic_gateway("192.168.50.30", "synthetic-other-gateway")]
    assert onboarding.home_connect(tmp_path)["gateways"][0]["connection_id"] != first_id
    current.clear()
    assert onboarding.home_connect(tmp_path)["status"] == "no_gateway_discovered"
    assert probed == ["192.168.50.20", "192.168.50.30", "192.168.50.30"]
    registry = json.loads((tmp_path / "gateways.private.json").read_text())
    assert all(not row["observed_this_run"] for row in registry["gateways"].values())


def test_multiple_gateways_keep_failures_separate_without_cloud_fallback(tmp_path, monkeypatch):
    candidates = [synthetic_gateway(), synthetic_gateway("192.168.50.30", "synthetic-other")]
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery(candidates, 2, 1))

    def inspect(candidate):
        if candidate == candidates[1]:
            return {"status": "local_api_unavailable", "error_code": "gateway_timeout"}
        return {"status": "read_only_connected", "inventory": {}, "topology": [], "inventory_counts": {}}

    monkeypatch.setattr(onboarding, "inspect_gateway", inspect)
    result = onboarding.home_connect(tmp_path)
    assert result["status"] == "partially_connected"
    assert result["gateways_connected"] == 1
    assert not result["discovery_complete"]
    assert result["gateways"][1]["error_code"] == "gateway_timeout"
    assert result["cloud_used"] is False
    assert result["app_used"] is False


@pytest.mark.parametrize(
    "other", [synthetic_gateway("192.168.50.30"), synthetic_gateway(did="synthetic-conflicting-identity")]
)
def test_conflicting_discovery_identities_are_not_probed(tmp_path, monkeypatch, other):
    monkeypatch.setattr(
        onboarding, "discover_gateways", lambda **_kwargs: Discovery([synthetic_gateway(), other], 1)
    )

    def forbidden_probe(_candidate):
        pytest.fail("conflicting discovery identity reached the TCP client")

    monkeypatch.setattr(onboarding, "inspect_gateway", forbidden_probe)
    result = onboarding.home_connect(tmp_path)
    assert result["gateways_connected"] == 0
    assert result["next_step"] == "resolve_gateway_identity_conflict"
    assert all(row["status"] == "identity_conflict" for row in result["gateways"])


def test_local_interface_failure_is_recorded_and_lock_is_released(tmp_path, monkeypatch):
    def no_interface(**_kwargs):
        raise DomainError("gateway_no_active_lan_interface")

    monkeypatch.setattr(onboarding, "discover_gateways", no_interface)
    result = onboarding.home_connect(tmp_path)
    assert result["status"] == "discovery_failed"
    assert result["error_code"] == "gateway_no_active_lan_interface"
    assert result["control_permission_granted"] is False
    assert not (tmp_path / "home-connect.lock").exists()


def test_discovered_but_closed_tcp_interface_does_not_claim_app_free_onboarding(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding, "discover_gateways", lambda **_kwargs: Discovery([synthetic_gateway()], 1)
    )
    # A bound socket without listen models an unavailable local-control endpoint,
    # without modifying the real gateway's setting or depending on an unused port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as closed_endpoint:
        closed_endpoint.bind(("127.0.0.1", 0))
        monkeypatch.setattr(
            onboarding,
            "YeelightReader",
            lambda _host, timeout: YeelightReader(*closed_endpoint.getsockname(), timeout=timeout),
        )
        result = onboarding.home_connect(tmp_path)
    assert result["status"] == "local_access_unavailable"
    assert result["gateways"][0]["status"] == "local_api_unavailable"
    assert result["gateways_connected"] == 0
    assert result["existing_local_api_connection_verified"] is False
    assert result["new_home_zero_touch_verified"] is False
    assert result["onboarding_requirements"]["automatic_local_control_enable_implemented"] is False
    assert result["physical_commands_sent"] == 0
    assert result["app_used"] is False


def test_gateway_not_discovered_is_not_reported_as_a_disabled_control_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(onboarding, "discover_gateways", lambda **_kwargs: Discovery([], 1))
    result = onboarding.home_connect(tmp_path)
    assert result["status"] == "no_gateway_discovered"
    assert result["gateways"] == []
    assert result["next_step"] == "check_same_lan_broadcast_and_local_control"
    assert result["new_home_zero_touch_verified"] is False


def test_invalid_private_registry_fails_before_network_without_echo(tmp_path, monkeypatch):
    (tmp_path / "gateways.private.json").write_text("PRIVATE_INVALID_REGISTRY", encoding="utf-8")
    monkeypatch.setattr(
        onboarding, "discover_gateways", lambda **_kwargs: pytest.fail("unexpected discovery")
    )
    with pytest.raises(DomainError) as error:
        onboarding.home_connect(tmp_path)
    assert error.value.code == "gateway_private_registry_invalid"
    assert "PRIVATE" not in str(error.value)
    assert not (tmp_path / "home-connect.lock").exists()


def test_runtime_inside_checkout_is_rejected_before_discovery(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(
        onboarding, "discover_gateways", lambda **_kwargs: pytest.fail("unexpected discovery")
    )
    with pytest.raises(DomainError, match="runtime_must_be_outside_checkout"):
        onboarding.home_connect(Path(__file__).resolve().parents[1] / "private-runtime")


def test_interface_detection_uses_up_connected_rfc1918_networks(monkeypatch):
    def row(address, mask="255.255.255.0", ptp=None):
        return SimpleNamespace(family=socket.AF_INET, address=address, netmask=mask, ptp=ptp)

    monkeypatch.setattr(
        discovery.psutil,
        "net_if_addrs",
        lambda: {
            "wired": [row("192.168.50.10")],
            "vpn": [row("10.20.0.1", ptp="10.20.0.2")],
            "down": [row("192.168.60.10")],
            "loopback": [row("127.0.0.1")],
            "benchmark": [row("198.18.0.1")],
            "bad-mask": [row("192.168.50.12", "0.0.0.0")],
        },
    )
    monkeypatch.setattr(
        discovery.psutil,
        "net_if_stats",
        lambda: {
            name: SimpleNamespace(isup=name != "down")
            for name in ("wired", "vpn", "down", "loopback", "benchmark", "bad-mask")
        },
    )
    assert discovery.lan_interfaces() == [LAN]
    assert discovery.lan_interfaces("192.168.50.10") == [LAN]
    with pytest.raises(DomainError, match="gateway_no_active_lan_interface"):
        discovery.lan_interfaces("8.8.8.8")


@pytest.mark.parametrize(
    "payload,source",
    [
        (advertisement("8.8.8.8"), "192.168.50.20"),
        (advertisement(), "8.8.8.8"),
        (advertisement("192.168.50.10"), "192.168.50.10"),
        (advertisement("192.168.50.255"), "192.168.50.255"),
        (advertisement() + b"ip:192.168.50.20\r\n", "192.168.50.20"),
        (b"not a discovery response", "192.168.50.20"),
        (b"\xff\xfe\r\n", "192.168.50.20"),
        (advertisement().replace(b"mac:", b"missing-mac:"), "192.168.50.20"),
        (b"x" * 4097, "192.168.50.20"),
    ],
)
def test_untrusted_discovery_cannot_redirect_connections(payload, source):
    assert discovery.parse_discovery(payload, source, LAN) is None


def test_discovery_accepts_modern_product_ids_without_assuming_capabilities():
    assert discovery.parse_discovery(advertisement(), "192.168.50.20", LAN) == synthetic_gateway()


def test_udp_discovery_uses_each_interface_and_deduplicates_replies(monkeypatch):
    targets = []
    received = []

    class Handler(socketserver.DatagramRequestHandler):
        def handle(self):
            received.append(self.rfile.read())
            assert received[-1] in discovery.PROBES
            # Lose the entire first probe round; later replies must still
            # discover one gateway without extending the overall read budget.
            if len(received) > len(discovery.PROBES):
                self.wfile.write(advertisement("127.0.0.2"))

    with socketserver.UDPServer(("127.0.0.2", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        real_socket = socket.socket

        class LoopbackProbeSocket(real_socket):
            def sendto(self, payload, target):
                targets.append(target)
                # Exercise real UDP framing without broadcasting onto CI/user LANs.
                return super().sendto(payload, server.server_address)

        monkeypatch.setattr(discovery.socket, "socket", LoopbackProbeSocket)
        monkeypatch.setattr(
            discovery,
            "lan_interfaces",
            lambda _address: [LanInterface("127.0.0.1", ipaddress.IPv4Network("127.0.0.0/8"))],
        )
        try:
            result = discovery.discover_gateways(timeout=0.2)
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert result.gateways == [synthetic_gateway("127.0.0.2")]
    assert result.interfaces_checked == 1 and result.interface_failures == 0
    assert 4 <= len(targets) <= 6
    assert all(target == ("127.255.255.255", discovery.DISCOVERY_PORT) for target in targets)
