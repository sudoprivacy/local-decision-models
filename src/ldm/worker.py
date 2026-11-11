import argparse
import json
import os
import traceback
from pathlib import Path

import psutil

from .contracts import DomainError
from .runtime import Runtime


def run_once(runtime: Runtime, stop_after_step: int | None = None) -> dict:
    job = runtime.claim()
    if job is None:
        return {"status": "idle"}
    try:
        from .training import train

        state = train(runtime, job, stop_after_step=stop_after_step)
        runtime.finish(job["id"], state)
        return {"run_id": job["id"], "status": state}
    except Exception as error:
        folder = runtime.directory(job["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        code = error.code if isinstance(error, DomainError) else "training_failed"
        runtime.finish(job["id"], "failed", code)
        return {"run_id": job["id"], "status": "failed", "error_code": code}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true", required=True)
    args = parser.parse_args()
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)
    result = run_once(Runtime(args.root))
    print(json.dumps(result))
    raise SystemExit(1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
