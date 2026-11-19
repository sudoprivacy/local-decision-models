"""Synthetic Yeelight gateway. Never pairs devices into the real mesh network."""

import copy
import json
import socketserver
import threading
from pathlib import Path

from .contracts import DomainError, HomeRequest
from .data import atomic_json
from .home import policy
from .yeelight import READS, YeelightReader, exchange


class MockGateway:
    def __init__(self):
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.online = True
        self.mock_commands_sent = 0
        self.nodes = [
            {
                "id": 101,
                "nt": 2,
                "n": "Synthetic study light",
                "pt": 3,
                "o": True,
                "params": {"p": False, "l": 30, "ct": 3000},
            },
            {"id": 102, "nt": 2, "n": "Synthetic presence sensor", "pt": 129, "o": True, "params": {}},
        ]
        self.rooms = [{"id": 201, "n": "Synthetic study"}]
        self.groups = [{"id": 301, "nt": 4, "n": "Synthetic reading lights", "pt": 3}]
        self.scenes = [{"id": 401, "n": "Synthetic reading", "params": {"state": "inactive"}}]
        gateway = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(0.5)
                try:
                    for _ in range(16):
                        line = self.rfile.readline(65537)
                        if not line or gateway.closed.is_set():
                            return
                        if len(line) > 65536 or not line.endswith(b"\r\n"):
                            return
                        response = gateway.respond(json.loads(line))
                        self.wfile.write((json.dumps(response) + "\r\n").encode("utf-8"))
                        self.wfile.flush()
                except (OSError, ValueError):
                    return

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.address = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def reader(self) -> YeelightReader:
        return YeelightReader(*self.address)

    def respond(self, request: dict) -> dict:
        if not isinstance(request, dict):
            return {"error": "invalid_request"}
        request_id, method = request.get("id"), request.get("method")
        with self.lock:
            if method in READS:
                field = READS[method]
                rows = getattr(self, field)
                selection = request.get("params", {}).get("id", 0)
                rows = [row for row in rows if not selection or row["id"] == selection]
                result = {"id": request_id, field: copy.deepcopy(rows)}
                if method == "gateway_get.topology":
                    result["method"] = "gateway_post.topology"
                return result
            if method != "gateway_set.prop":
                return {"id": request_id, "error": "unsupported_mock_method"}
            nodes = request.get("nodes")
            if not isinstance(nodes, list) or len(nodes) != 1 or not isinstance(nodes[0], dict):
                return {"id": request_id, "error": "unsupported_mock_target"}
            target = nodes[0]
            values = target.get("set")
            if target.get("id") != 101 or target.get("nt") != 2 or not isinstance(values, dict):
                return {"id": request_id, "error": "unsupported_mock_target"}
            if not values or set(values) - {"p", "l", "ct"}:
                return {"id": request_id, "error": "unsupported_mock_property"}
            if (
                "p" in values
                and type(values["p"]) is not bool
                or "l" in values
                and (type(values["l"]) is not int or not 1 <= values["l"] <= 100)
                or "ct" in values
                and (type(values["ct"]) is not int or not 2700 <= values["ct"] <= 6500)
            ):
                return {"id": request_id, "error": "invalid_mock_property"}
            if not self.online:
                return {"id": request_id, "error": "mock_device_offline"}
            self.nodes[0]["params"].update(values)
            self.mock_commands_sent += 1
            return {"id": request_id, "result": "ok"}

    def set_online(self, online: bool):
        with self.lock:
            self.online = online
            self.nodes[0]["o"] = online

    def reading(self, request: HomeRequest, role: str, probabilities: list[float] | None) -> dict:
        """Only the synthetic study fixture is wired to the existing approval policy."""
        decision = policy(request, role, probabilities)
        if decision["status"] != "simulated":
            return decision
        if request.space != "study":
            return {**decision, "status": "denied", "reason": "mock_space_not_configured"}
        if not self.reader().query("gateway_get.node", 101)[0]["o"]:
            return {**decision, "status": "manual_fallback", "reason": "mock_device_offline"}
        expected = {"p": True, "l": 70, "ct": 4000}
        # A write endpoint comes only from this owned loopback server. The live
        # YeelightReader exposes no write path; no host is accepted from a request.
        exchange(
            *self.address,
            {
                "id": 2,
                "method": "gateway_set.prop",
                "nodes": [{"id": 101, "nt": 2, "set": expected}],
            },
        )
        actual = self.reader().query("gateway_get.node", 101)[0]["params"]
        if actual != expected:
            raise DomainError("mock_state_readback_mismatch")
        return {**decision, "mock_commands_sent": self.mock_commands_sent, "state": actual}


def home_mock(root: Path) -> dict:
    root = root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise DomainError("runtime_must_be_outside_checkout")
    with MockGateway() as gateway:
        initial = gateway.reader().inventory()
        scenarios = []
        for role, payload, expected in [
            ("owner", {"intent": "reading"}, "confirmation_required"),
            ("guest", {"intent": "reading", "human_confirmed": True}, "denied"),
            ("owner", {"intent": "ambiguous"}, "clarify"),
            ("owner", {"intent": "reading", "model_online": False}, "manual_fallback"),
            ("owner", {"intent": "open_entry", "human_confirmed": True}, "denied"),
        ]:
            response = gateway.reading(HomeRequest(**payload), role, [0.95, 0.03, 0.02])
            if response["status"] != expected or gateway.mock_commands_sent:
                raise DomainError("mock_authorization_regression")
            scenarios.append({"role": role, "request": payload, "result": response})
        confirmed = HomeRequest(intent="reading", human_confirmed=True)
        applied = gateway.reading(confirmed, "owner", [0.95, 0.03, 0.02])
        gateway.set_online(False)
        offline = gateway.reading(confirmed, "owner", [0.95, 0.03, 0.02])
        if gateway.mock_commands_sent != 1 or offline["status"] != "manual_fallback":
            raise DomainError("mock_offline_regression")
        receipt = {
            "evidence_level": "synthetic_gateway_protocol",
            "decision_source": "fixed_synthetic_distribution",
            "gateway": "loopback_yeelight_mock",
            "inventory_counts": {name: len(rows) for name, rows in initial.items()},
            "blocked_or_deferred": scenarios,
            "confirmed_reading": applied,
            "offline": offline,
            "mock_commands_sent": gateway.mock_commands_sent,
            "physical_commands_sent": 0,
            "model_quality_established": False,
            "physical_device_acceptance": False,
        }
    atomic_json(root / "home-mock-receipt.json", receipt)
    return receipt
