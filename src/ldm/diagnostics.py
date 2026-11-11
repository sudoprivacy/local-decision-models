import json

from .contracts import Case, Diagnostics, DomainError, GroupDiagnostic, Metric
from .training import MODEL_ID

METRICS = {"mae", "points_weighted_mae", "within_5pp", "within_10pp", "accuracy", "brier"}
RESOURCES = {"cpu_threads", "elapsed_seconds", "rss_bytes"}


def metric_rows(value: dict) -> list[Metric]:
    return [Metric(name=name, value=value[name]) for name in sorted(METRICS & value.keys())]


def diagnostics(runtime, run_id: str, view: str = "overall") -> Diagnostics:
    if view not in {"overall", "by_group"}:
        raise DomainError("view_not_approved")
    job = runtime.job(run_id)
    path = runtime.directory(run_id) / "diagnostics.private.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    groups, suppressed = [], 0
    if view == "by_group":
        for index, group in enumerate(raw.get("groups", [])):
            if group["count"] < 8:
                suppressed += 1
            else:
                groups.append(
                    GroupDiagnostic(
                        group=f"approved-group-{index + 1}",
                        count=group["count"],
                        metrics=metric_rows(group["metrics"]),
                    )
                )
    return Diagnostics(
        run_id=run_id,
        status=job["state"],
        case=Case(job["case_name"]),
        evidence_level="synthetic_cpu_smoke",
        dataset_version=job["version"],
        split_id=job["request"]["split_id"],
        recipe_id=job["request"]["recipe_id"],
        code_version=job["code_version"],
        model_id=MODEL_ID,
        objective=raw.get("objective", "not_started"),
        steps=raw.get("steps", 0),
        validation_count=raw.get("validation_count", 0),
        independent_validation_groups=raw.get("independent_validation_groups", 0),
        train_curve=raw.get("train_curve", []),
        validation=metric_rows(raw.get("validation", {})),
        groups=groups,
        suppressed_groups=suppressed,
        resource={name: value for name, value in raw.get("resource", {}).items() if name in RESOURCES},
        error_code=job["error_code"],
        claims=[
            "Synthetic CPU execution evidence; domain quality and target-device acceptance remain open.",
            "Physics candidate weights are a decision readout, not an observed student-score histogram.",
            "Supervised objective; no RLCD reproduction or gain claim.",
            "Final-test labels were not used by the training or validation loop.",
        ],
    )


def compare(runtime, run_ids: list[str]) -> dict:
    jobs = [runtime.job(run_id) for run_id in run_ids]

    def key(job):
        return (
            job["case_name"],
            job["version"],
            job["request"]["split_id"],
            job["request"]["seed"],
            job["request"]["max_steps"],
            job["request"]["max_seconds"],
            job["code_version"],
        )

    if any(job["state"] != "completed" for job in jobs):
        raise DomainError("completed_runs_required")
    if any(key(job) != key(jobs[0]) for job in jobs[1:]):
        raise DomainError("runs_not_comparable")
    reports = [diagnostics(runtime, run_id) for run_id in run_ids]
    first = {metric.name: metric.value for metric in reports[0].validation}
    return {
        "run_ids": run_ids,
        "comparable": True,
        "evidence_level": "synthetic_cpu_smoke",
        "same_budget": True,
        "deltas_from_first": [
            {metric.name: metric.value - first[metric.name] for metric in report.validation}
            for report in reports[1:]
        ],
        "uncertainty": "insufficient_independent_validation_groups",
        "domain_quality_conclusion": "not_established",
    }
