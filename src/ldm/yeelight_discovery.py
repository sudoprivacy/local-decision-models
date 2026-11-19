"""Bounded UDP discovery on connected LANs; no cloud login or device changes."""

import ipaddress
import re
import selectors
import socket
import time
from dataclasses import dataclass

import psutil

from .contracts import DomainError

DISCOVERY_PORT = 1982
PROBES = (b"YEELIGHT_GATEWAY_CONTROL_DISCOVER", b"YEELIGHT_DEVICE_CONTROL_DISCOVER")
PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
MAX_INTERFACES = 16
MAX_GATEWAYS = 16
MAX_DATAGRAMS = 256


@dataclass(frozen=True)
class LanInterface:
    address: str
    network: ipaddress.IPv4Network


@dataclass(frozen=True)
class Gateway:
    host: str
    mac: str
    did: str
    pid: str

    @property
    def identity(self) -> str:
        # Discovery identities are hints, never authentication or control grants.
        return self.mac + "/" + self.did


@dataclass
class Discovery:
    gateways: list[Gateway]
    interfaces_checked: int
    interface_failures: int = 0


def lan_interfaces(address: str | None = None) -> list[LanInterface]:
    try:
        stats = psutil.net_if_stats()
        addresses = psutil.net_if_addrs()
    except OSError:
        raise DomainError("gateway_interface_inspection_failed") from None
    found = {}
    for name, rows in addresses.items():
        if name not in stats or not stats[name].isup:
            continue
        for row in rows:
            if row.family != socket.AF_INET or not row.netmask or row.ptp:
                continue
            try:
                interface = ipaddress.IPv4Interface(f"{row.address}/{row.netmask}")
            except ValueError:
                continue
            if interface.network.prefixlen > 30 or not any(
                interface.network.subnet_of(private) for private in PRIVATE_NETWORKS
            ):
                continue
            if interface.ip in (interface.network.network_address, interface.network.broadcast_address):
                continue
            if address is None or str(interface.ip) == address:
                found[str(interface.ip)] = LanInterface(str(interface.ip), interface.network)
    if not found:
        raise DomainError("gateway_no_active_lan_interface")
    if len(found) > MAX_INTERFACES:
        raise DomainError("gateway_too_many_interfaces_select_one")
    return [found[key] for key in sorted(found)]


def parse_discovery(payload: bytes, source: str, interface: LanInterface) -> Gateway | None:
    if len(payload) > 4096:
        return None
    try:
        sender = ipaddress.IPv4Address(source)
        if sender not in interface.network or sender in (
            interface.network.network_address,
            interface.network.broadcast_address,
            ipaddress.IPv4Address(interface.address),
        ):
            return None
        fields = {}
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key in fields:
                return None
            fields[key] = value.strip()
        # Use the actual datagram sender, never a different advertised target.
        if ipaddress.IPv4Address(fields["ip"]) != sender:
            return None
        mac = fields["mac"].replace("-", ":").upper()
        if not re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", mac):
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", fields["did"]):
            return None
        if not re.fullmatch(r"[0-9]{1,10}", fields["pid"]) or int(fields["pid"]) == 0:
            return None
        return Gateway(str(sender), mac, fields["did"], fields["pid"])
    except (KeyError, ValueError, UnicodeError):
        return None


def discover_gateways(interface: str | None = None, timeout: float = 3.0) -> Discovery:
    if not 0 < timeout <= 5:
        raise DomainError("gateway_invalid_discovery_budget")
    interfaces = lan_interfaces(interface)
    gateways, failures, datagrams = set(), 0, 0
    connections = []
    try:
        with selectors.DefaultSelector() as selector:
            for lan in interfaces:
                connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                connections.append(connection)
                try:
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    connection.bind((lan.address, 0))
                    connection.setblocking(False)
                    for probe in PROBES:
                        connection.sendto(probe, (str(lan.network.broadcast_address), DISCOVERY_PORT))
                    selector.register(connection, selectors.EVENT_READ, lan)
                except OSError:
                    failures += 1
                    connection.close()
            if not selector.get_map():
                raise DomainError("gateway_discovery_unavailable")
            started = time.monotonic()
            deadline = started + timeout
            interval, retries_left = timeout / 3, 2
            next_probe = started + interval
            while (remaining := deadline - time.monotonic()) > 0:
                if retries_left and time.monotonic() >= next_probe:
                    # UDP discovery may lose replies. Retry reads only, within
                    # the original total deadline; never retry device writes.
                    for key in list(selector.get_map().values()):
                        try:
                            for probe in PROBES:
                                key.fileobj.sendto(
                                    probe, (str(key.data.network.broadcast_address), DISCOVERY_PORT)
                                )
                        except OSError:
                            failures += 1
                            selector.unregister(key.fileobj)
                    retries_left -= 1
                    next_probe += interval
                wait = min(remaining, max(0, next_probe - time.monotonic())) if retries_left else remaining
                for key, _events in selector.select(wait):
                    try:
                        payload, source = key.fileobj.recvfrom(4097)
                    except BlockingIOError:
                        continue
                    except OSError:
                        failures += 1
                        selector.unregister(key.fileobj)
                        continue
                    datagrams += 1
                    if datagrams > MAX_DATAGRAMS:
                        raise DomainError("gateway_discovery_budget_exceeded")
                    candidate = parse_discovery(payload, source[0], key.data)
                    if candidate is not None:
                        gateways.add(candidate)
                        if len(gateways) > MAX_GATEWAYS:
                            raise DomainError("gateway_discovery_budget_exceeded")
                if not selector.get_map():
                    break
    except OSError:
        raise DomainError("gateway_discovery_unavailable") from None
    finally:
        for connection in connections:
            connection.close()
    return Discovery(sorted(gateways, key=lambda item: (item.identity, item.host)), len(interfaces), failures)
