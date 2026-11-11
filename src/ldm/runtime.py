import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import psutil

from .contracts import Case, DomainError, RunRequest
from .data import Catalog, atomic_json, digest, fixture_records, physics_labels


def code_version() -> str:
    source = Path(__file__).parent
    result = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        path = source.parents[1] / name
        if path.is_file():
            result.update(name.encode())
            result.update(path.read_bytes())
    return "source-" + result.hexdigest()[:24]


def bootstrap(root: Path, physics_csv: Path | None = None) -> dict:
    root = root.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if root.is_relative_to(checkout):
        raise DomainError("runtime_must_be_outside_checkout")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "catalog.json").exists():
        raise DomainError("runtime_already_initialized")
    catalog = {}
    for case in Case:
        dataset_id = "fixture-physics-v1" if case == Case.PHYSICS else "fixture-home-v1"
        path = root / "datasets" / (dataset_id + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "\n".join(row.model_dump_json() for row in fixture_records(case)) + "\n", encoding="utf-8"
        )
        catalog[dataset_id] = {
            "case": case.value,
            "kind": "records",
            "privacy": "synthetic",
            "version": dataset_id,
            "relative_path": str(path.relative_to(root)),
            "sha256": digest(path),
        }
    if physics_csv is not None:
        physics_csv = physics_csv.resolve(strict=True)
        _, coverage = physics_labels(physics_csv)
        expected = [
            (2020, 30, 93),
            (2021, 31, 92),
            (2022, 30, 94),
            (2023, 33, 94),
            (2024, 31, 94),
            (2025, 32, 91),
            (2026, 32, 95),
        ]
        if [
            (row["year"], row["observed_rows"], row["covered_points"]) for row in coverage["years"]
        ] != expected:
            raise DomainError("physics_pilot_coverage_version_mismatch")
        catalog["physics-xicheng-2020-2026-v1"] = {
            "case": Case.PHYSICS.value,
            "kind": "physics_labels_only",
            "privacy": "private",
            "version": "gaokao-xicheng-2020-2026-received-20261102-v1",
            "private_path": str(physics_csv),
            "sha256": digest(physics_csv),
            "coverage": coverage,
        }
    atomic_json(root / "catalog.json", catalog)
    credentials = {role: secrets.token_urlsafe(32) for role in ("owner", "researcher", "guest")}
    atomic_json(root / "credentials.json", credentials)
    if os.name != "nt":
        (root / "credentials.json").chmod(0o600)
        root.chmod(0o700)
    Runtime(root)
    return {
        "root": str(root),
        "datasets": list(catalog),
        "credentials": "credentials.json (local only)",
        "mode": "local_single_operator_reference",
        "service_identity_separation": "deployment_required",
    }


class Runtime:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.catalog = Catalog(self.root)
        with self.db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, request TEXT NOT NULL,
                    case_name TEXT NOT NULL, version TEXT NOT NULL, code_version TEXT NOT NULL,
                    created REAL NOT NULL, stop_requested INTEGER NOT NULL DEFAULT 0,
                    worker_identity TEXT, error_code TEXT
                );
                CREATE TABLE IF NOT EXISTS shadow_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, case_name TEXT NOT NULL,
                    run_id TEXT NOT NULL, action TEXT NOT NULL, approved_by TEXT NOT NULL,
                    note TEXT NOT NULL, created REAL NOT NULL
                );
            """)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.root / "experiments.sqlite", timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def credentials(self) -> dict[str, str]:
        return json.loads((self.root / "credentials.json").read_text(encoding="utf-8"))

    def job(self, run_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise DomainError("invalid_run_id")
        with self.db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise DomainError("run_not_found")
        result = dict(row)
        result["request"] = json.loads(result["request"])
        return result

    def directory(self, run_id: str) -> Path:
        self.job(run_id)
        return self.root / "runs" / run_id

    def enqueue(self, request: RunRequest) -> dict:
        item = self.catalog.item(request.dataset_id)
        if item["privacy"] != "synthetic":
            raise DomainError("real_training_recipe_not_accepted")
        self.catalog.records(request.dataset_id)
        self.catalog.load_split(request.split_id, request.dataset_id)
        run_id = uuid.uuid4().hex
        with self.db() as conn:
            conn.execute(
                "INSERT INTO jobs(id,state,request,case_name,version,code_version,created) VALUES(?,?,?,?,?,?,?)",
                (
                    run_id,
                    "queued",
                    request.model_dump_json(),
                    item["case"],
                    item["version"],
                    code_version(),
                    time.time(),
                ),
            )
        return {"run_id": run_id, "status": "queued", "evidence_level": "synthetic_cpu_smoke"}

    def claim(self) -> dict | None:
        with self.db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                return None
            row = conn.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:
                return None
            identity = json.dumps({"pid": os.getpid(), "birth": psutil.Process().create_time()})
            conn.execute(
                "UPDATE jobs SET state='running',worker_identity=?,stop_requested=0 WHERE id=?",
                (identity, row["id"]),
            )
        return self.job(row["id"])

    def finish(self, run_id: str, state: str, error_code: str | None = None):
        if state not in {"completed", "stopped", "failed"}:
            raise DomainError("invalid_state_transition")
        with self.db() as conn:
            if (
                conn.execute(
                    "UPDATE jobs SET state=?,error_code=?,worker_identity=NULL WHERE id=? AND state='running'",
                    (state, error_code, run_id),
                ).rowcount
                != 1
            ):
                raise DomainError("run_not_running")

    def stop(self, run_id: str) -> dict:
        self.job(run_id)
        with self.db() as conn:
            conn.execute(
                "UPDATE jobs SET stop_requested=1, state=CASE WHEN state='queued' THEN 'stopped' ELSE state END WHERE id=?",
                (run_id,),
            )
        return {"run_id": run_id, "status": self.job(run_id)["state"], "stop_requested": True}

    def resume(self, run_id: str) -> dict:
        job = self.job(run_id)
        if job["state"] == "running":
            identity = json.loads(job["worker_identity"])
            try:
                if psutil.Process(identity["pid"]).create_time() == identity["birth"]:
                    raise DomainError("worker_still_running")
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                raise DomainError("worker_identity_unknown") from None
        elif job["state"] != "stopped":
            raise DomainError("run_not_resumable")
        self.catalog.records(job["request"]["dataset_id"])
        self.catalog.load_split(job["request"]["split_id"], job["request"]["dataset_id"])
        with self.db() as conn:
            conn.execute(
                "UPDATE jobs SET state='queued',stop_requested=0,worker_identity=NULL,error_code=NULL WHERE id=? AND state=?",
                (run_id, job["state"]),
            )
        return {"run_id": run_id, "status": "queued"}

    def verify_artifacts(self, run_id: str) -> dict:
        folder = self.directory(run_id)
        manifest_path = folder / "artifacts.json"
        if not manifest_path.is_file():
            raise DomainError("checkpoint_or_evaluation_missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {"checkpoint.pt", "diagnostics.private.json"}
        if set(manifest) != expected:
            raise DomainError("artifact_manifest_invalid")
        if any(digest(folder / name) != sha for name, sha in manifest.items()):
            raise DomainError("artifact_integrity_failed")
        return manifest

    def shadow(self, run_id: str, note: str, rollback: bool = False) -> dict:
        job = self.job(run_id)
        if job["state"] != "completed":
            raise DomainError("completed_evaluation_required")
        self.verify_artifacts(run_id)
        with self.db() as conn:
            if (
                rollback
                and not conn.execute(
                    "SELECT 1 FROM shadow_events WHERE run_id=? AND action='approve_shadow'", (run_id,)
                ).fetchone()
            ):
                raise DomainError("rollback_target_not_previously_approved")
            conn.execute(
                "INSERT INTO shadow_events(case_name,run_id,action,approved_by,note,created) VALUES(?,?,?,?,?,?)",
                (
                    job["case_name"],
                    run_id,
                    "rollback" if rollback else "approve_shadow",
                    "owner",
                    note,
                    time.time(),
                ),
            )
        return {
            "run_id": run_id,
            "channel": "synthetic_shadow",
            "approved_by": "owner",
            "hardware_control": False,
            "action": "rollback" if rollback else "approve_shadow",
        }

    def active_shadow(self, case: Case) -> str | None:
        with self.db() as conn:
            row = conn.execute(
                "SELECT run_id FROM shadow_events WHERE case_name=? ORDER BY seq DESC LIMIT 1", (case.value,)
            ).fetchone()
        return row["run_id"] if row else None
