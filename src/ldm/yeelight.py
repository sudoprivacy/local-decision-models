"""Bounded, read-only Yeelight Pro LAN protocol client.

Inventory stays local. The research HTTP API does not accept gateway addresses.
The protocol subset follows the owner-supplied Yeelight Pro LAN PDF v2.22.
"""

import ipaddress
import json
import socket
import time

from .contracts import DomainError

READS = {
    "gateway_get.topology": "nodes",
    "gateway_get.node": "nodes",
    "gateway_get.room": "rooms",
    "gateway_get.group": "groups",
    "gateway_get.scene": "scenes",
}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def exchange(host: str, port: int, request: dict, timeout: float = 3.0) -> dict:
    """Internal framing shared with the loopback-only mock; no automatic retries."""
    deadline = time.monotonic() + timeout
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            connection.sendall((json.dumps(request, allow_nan=False) + "\r\n").encode("utf-8"))
            buffered = b""
            received = 0
            while time.monotonic() < deadline:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                incoming = connection.recv(min(65536, MAX_RESPONSE_BYTES - received + 1))
                if not incoming:
                    raise DomainError("gateway_connection_closed")
                received += len(incoming)
                if received > MAX_RESPONSE_BYTES:
                    raise DomainError("gateway_response_too_large")
                buffered += incoming
                while b"\r\n" in buffered:
                    line, buffered = buffered.split(b"\r\n", 1)
                    if not line:
                        continue
                    response = json.loads(line)
                    if not isinstance(response, dict):
                        raise DomainError("gateway_invalid_response")
                    if response.get("id") != request["id"]:
                        continue
                    if "error" in response or response.get("result") not in (None, "ok"):
                        raise DomainError("gateway_request_rejected")
                    return response
    except TimeoutError:
        raise DomainError("gateway_timeout") from None
    except (OSError, UnicodeError, ValueError):
        raise DomainError("gateway_protocol_or_connection_error") from None
    raise DomainError("gateway_timeout")


class YeelightReader:
    """The live adapter permits inventory reads only, including on loopback."""

    def __init__(self, host: str, port: int = 65443, timeout: float = 3.0):
        try:
            address = ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError:
            raise DomainError("gateway_private_ipv4_required") from None
        if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
            raise DomainError("gateway_private_ipv4_required")
        if not 1 <= port <= 65535 or not 0 < timeout <= 10:
            raise DomainError("gateway_invalid_connection_budget")
        self.host, self.port, self.timeout = str(address), port, timeout

    def query(self, method: str, node_id: int = 0) -> list[dict]:
        if method not in READS:
            raise DomainError("gateway_read_only")
        if type(node_id) is not int or node_id < 0:
            raise DomainError("gateway_invalid_node_id")
        request = {"id": 1, "method": method}
        if method != "gateway_get.topology":
            request["params"] = {"id": node_id}
        response = exchange(self.host, self.port, request, self.timeout)
        rows = response.get(READS[method])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise DomainError("gateway_invalid_inventory")
        if method == "gateway_get.topology" and response.get("method") != "gateway_post.topology":
            raise DomainError("gateway_invalid_topology")
        return rows

    def inventory(self) -> dict[str, list[dict]]:
        return {
            collection: self.query("gateway_get." + resource)
            for resource, collection in (
                ("node", "nodes"),
                ("room", "rooms"),
                ("group", "groups"),
                ("scene", "scenes"),
            )
        }

    def summary(self) -> dict:
        return {
            "mode": "read_only_gateway_inventory",
            "counts": {name: len(rows) for name, rows in self.inventory().items()},
            "physical_commands_sent": 0,
            "device_control_acceptance": False,
        }
