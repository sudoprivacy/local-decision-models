import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .contracts import Case, DomainError
from .data import atomic_json
from .runtime import Runtime, bootstrap


def invoke_worker(root: Path) -> dict:
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "ldm.worker", "--root", str(root), "--once"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise DomainError("worker_failed_check_local_receipt")
    return json.loads(result.stdout.strip().splitlines()[-1])


def smoke(root: Path) -> dict:
    """Runs both cases through the same authenticated API and separate workers."""
    from fastapi.testclient import TestClient

    from .api import create_app

    runtime = Runtime(root)
    tokens = runtime.credentials()
    with TestClient(create_app(root)) as client:
        researcher = {"Authorization": "Bearer " + tokens["researcher"]}
        owner = {"Authorization": "Bearer " + tokens["owner"]}

        def call(method, path, payload=None, headers=None):
            response = client.request(method, path, json=payload, headers=headers or researcher)
            if response.status_code >= 400:
                raise DomainError(response.json().get("error_code", "smoke_request_failed"))
            return response.json()

        results = []
        for case, dataset in ((Case.PHYSICS, "fixture-physics-v1"), (Case.HOME, "fixture-home-v1")):
            prepared = call("POST", "/v1/datasets/prepare", {"dataset_id": dataset})
            split = call("POST", "/v1/splits", {"dataset_id": dataset})
            run_ids = []
            for recipe in ("cpu-head-v1", "cpu-lora-v1"):
                run = call(
                    "POST",
                    "/v1/experiments",
                    {
                        "dataset_id": dataset,
                        "split_id": split["split_id"],
                        "recipe_id": recipe,
                        "max_steps": 8,
                        "max_seconds": 120,
                        "seed": 7,
                    },
                )
                if invoke_worker(root)["status"] != "completed":
                    raise DomainError("smoke_worker_not_completed")
                run_ids.append(run["run_id"])
            comparison = call("POST", "/v1/experiments/compare", {"run_ids": run_ids})
            for run_id in run_ids:
                call(
                    "POST",
                    f"/v1/experiments/{run_id}/shadow",
                    {"note": "Synthetic smoke approval only"},
                    owner,
                )
            call(
                "POST",
                f"/v1/experiments/{run_ids[0]}/rollback",
                {"note": "Synthetic rollback exercise"},
                owner,
            )
            results.append(
                {
                    "case": case.value,
                    "prepared": prepared,
                    "runs": run_ids,
                    "diagnostics": call("GET", f"/v1/experiments/{run_ids[1]}/diagnostics?view=by_group"),
                    "comparison": comparison,
                    "shadow_rollback": "passed",
                }
            )
        replay = []
        for role, request in [
            ("owner", {"intent": "reading", "human_confirmed": True}),
            ("guest", {"intent": "reading"}),
            ("owner", {"intent": "ambiguous"}),
            ("owner", {"intent": "reading", "model_online": False}),
            ("owner", {"intent": "open_entry", "human_confirmed": True}),
        ]:
            replay.append(
                call("POST", "/v1/home/preview", request, {"Authorization": "Bearer " + tokens[role]})
            )
        if any(row["physical_commands_sent"] for row in replay):
            raise DomainError("simulator_boundary_failed")
        receipt = {
            "evidence_level": "synthetic_cpu_smoke",
            "case_results": results,
            "home_replay": replay,
            "real_model_quality_established": False,
            "target_gpu_verified": False,
            "physical_devices_connected": False,
        }
        atomic_json(root / "smoke-receipt.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description="Local decision-model reference service")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--physics-csv", type=Path)
    for name in ("smoke", "serve", "readiness"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = bootstrap(args.root, args.physics_csv)
        elif args.command == "smoke":
            result = smoke(args.root)
        elif args.command == "readiness":
            runtime = Runtime(args.root)
            result = {"datasets": [runtime.catalog.prepare(key) for key in runtime.catalog.items]}
        else:
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(args.root), host="127.0.0.1", port=args.port, access_log=False)
            return
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    except DomainError as error:
        print(json.dumps({"error_code": error.code}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
