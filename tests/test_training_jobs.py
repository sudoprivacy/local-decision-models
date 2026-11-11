import json

import pytest
import torch

from ldm.contracts import Case, DomainError, RunRequest
from ldm.diagnostics import compare, diagnostics
from ldm.runtime import Runtime
from ldm.training import DecisionModel, batch, evaluate, objective
from ldm.worker import run_once


def queued(runtime, dataset="fixture-home-v1", recipe="cpu-lora-v1", steps=4):
    split = runtime.catalog.split(dataset)
    return runtime.enqueue(
        RunRequest(dataset_id=dataset, split_id=split["split_id"], recipe_id=recipe, max_steps=steps)
    )


def test_lora_updates_only_adapter_and_decision_head(runtime):
    records = runtime.catalog.records("fixture-home-v1")[:8]
    model = DecisionModel(Case.HOME, "cpu-lora-v1", 7)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable and all("lora_" in name or name.startswith("head.") for name in trainable)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=0.01)
    loss = objective(model(*batch(records)), records)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before[name], value) for name, value in model.named_parameters() if "lora_" in name
    )
    assert all(
        torch.equal(before[name], value) for name, value in model.named_parameters() if name not in trainable
    )


def test_physics_reports_mean_error_not_confidence(runtime):
    records = runtime.catalog.records("fixture-physics-v1")[:2]
    records[0].mean_score_rate, records[0].points = 1.0, 1.0
    records[1].mean_score_rate, records[1].points = 0.5, 3.0
    metrics = evaluate(records, [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
    assert metrics["mae"] == 0.5
    assert metrics["points_weighted_mae"] == 0.25
    assert metrics["within_5pp"] == 0.5


def test_checkpoint_stop_resume_matches_uninterrupted_training(runtime):
    one = queued(runtime)
    assert run_once(runtime)["status"] == "completed"
    two = queued(runtime)
    assert run_once(runtime, stop_after_step=2)["status"] == "stopped"
    restarted = Runtime(runtime.root)
    restarted.resume(two["run_id"])
    assert run_once(restarted)["status"] == "completed"
    left = torch.load(runtime.directory(one["run_id"]) / "checkpoint.pt", weights_only=True)
    right = torch.load(runtime.directory(two["run_id"]) / "checkpoint.pt", weights_only=True)
    assert left["step"] == right["step"] == 4
    assert left["curve"] == right["curve"]
    assert all(torch.equal(value, right["model"][name]) for name, value in left["model"].items())


def test_stop_before_first_step_is_resumable(runtime):
    run = queued(runtime, steps=1)
    assert run_once(runtime, stop_after_step=0)["status"] == "stopped"
    runtime.resume(run["run_id"])
    assert run_once(runtime)["status"] == "completed"


def test_checkpoint_tampering_is_not_resumed(runtime):
    run = queued(runtime)
    assert run_once(runtime, stop_after_step=1)["status"] == "stopped"
    (runtime.directory(run["run_id"]) / "checkpoint.pt").write_bytes(b"tampered")
    runtime.resume(run["run_id"])
    result = run_once(runtime)
    assert result["status"] == "failed"
    assert result["error_code"] == "checkpoint_integrity_failed"


def test_api_enqueues_and_cannot_publish_unevaluated_model(runtime, client, auth):
    split = runtime.catalog.split("fixture-physics-v1")
    result = client.post(
        "/v1/experiments",
        headers=auth["researcher"],
        json={
            "dataset_id": "fixture-physics-v1",
            "split_id": split["split_id"],
            "recipe_id": "cpu-head-v1",
            "max_steps": 1,
        },
    )
    assert result.status_code == 202
    run_id = result.json()["run_id"]
    assert runtime.job(run_id)["state"] == "queued"
    assert not (runtime.directory(run_id) / "checkpoint.pt").exists()
    assert (
        client.post(
            f"/v1/experiments/{run_id}/shadow", headers=auth["owner"], json={"note": "test"}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/v1/experiments/{run_id}/shadow", headers=auth["researcher"], json={"note": "test"}
        ).status_code
        == 403
    )


def test_diagnostics_project_only_approved_fields_and_suppress_small_groups(runtime):
    run = queued(runtime, steps=1)
    assert run_once(runtime)["status"] == "completed"
    path = runtime.directory(run["run_id"]) / "diagnostics.private.json"
    raw = json.loads(path.read_text())
    raw["raw_label"] = "SECRET_RECORD"
    raw["groups"] = [{"group": "SECRET_NAME", "count": 1, "metrics": {"accuracy": 1}}]
    raw["resource"]["SECRET_PATH"] = 123
    path.write_text(json.dumps(raw), encoding="utf-8")
    public = diagnostics(runtime, run["run_id"], "by_group")
    assert public.suppressed_groups == 1
    assert not public.groups
    assert "SECRET" not in public.model_dump_json()
    assert "private_probabilities" not in public.model_dump_json()
    with pytest.raises(DomainError, match="artifact_integrity_failed"):
        runtime.shadow(run["run_id"], "cannot approve changed evidence")


def test_equal_budget_comparison_and_rollback_require_evidence(runtime):
    first = queued(runtime, recipe="cpu-head-v1", steps=1)
    run_once(runtime)
    second = queued(runtime, steps=1)
    run_once(runtime)
    report = compare(runtime, [first["run_id"], second["run_id"]])
    assert report["comparable"]
    assert report["uncertainty"] == "insufficient_independent_validation_groups"
    with pytest.raises(DomainError, match="rollback_target_not_previously_approved"):
        runtime.shadow(first["run_id"], "not published yet", rollback=True)
    runtime.shadow(first["run_id"], "synthetic owner approval")
    runtime.shadow(second["run_id"], "synthetic owner approval")
    runtime.shadow(first["run_id"], "restore previous", rollback=True)
    assert runtime.active_shadow(Case.HOME) == first["run_id"]


def test_worker_errors_stay_private(runtime, monkeypatch):
    queued_run = queued(runtime, steps=1)

    def private_error(*_args, **_kwargs):
        raise RuntimeError("SECRET_RAW_LABEL")

    monkeypatch.setattr("ldm.training.train", private_error)
    result = run_once(runtime)
    assert result["error_code"] == "training_failed"
    assert "SECRET" not in diagnostics(runtime, queued_run["run_id"]).model_dump_json()
    assert "SECRET" in (runtime.directory(queued_run["run_id"]) / "error.private.log").read_text()
