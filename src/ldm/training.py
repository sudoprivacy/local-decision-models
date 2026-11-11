import time
from collections import defaultdict

import psutil
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from .contracts import Case, DomainError, Record
from .data import atomic_json, canonical_hash, digest

MODEL_ID = "tiny-random-qwen3-v1"
MODEL_VERSION = "qwen3-config-32h-1layer-259vocab-v1"


class DecisionModel(nn.Module):
    """Real tensor/PEFT smoke backend. No pretrained weights or domain-quality claim."""

    def __init__(self, case: Case, recipe_id: str, seed: int):
        super().__init__()
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=259,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=1024,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "eager"
        backbone = Qwen3Model(config)
        backbone.requires_grad_(False)
        if recipe_id == "cpu-lora-v1":
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none"
                ),
            )
        elif recipe_id != "cpu-head-v1":
            raise DomainError("recipe_not_registered")
        self.backbone = backbone
        self.case = case
        torch.manual_seed(seed)
        self.head = nn.Linear(config.hidden_size, 5 if case == Case.PHYSICS else 3)

    def forward(self, ids, mask):
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.sum(dim=1) - 1
        return self.head(hidden[torch.arange(len(ids)), last])


def batch(records: list[Record]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [[1, *(value + 3 for value in row.prompt().encode("utf-8")), 2] for row in records]
    length = max(map(len, tokens))
    if length > 1024:
        raise DomainError("full_context_exceeds_registered_limit")
    ids = torch.zeros((len(tokens), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(tokens):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1
    return ids, mask


def objective(logits: torch.Tensor, records: list[Record]) -> torch.Tensor:
    if records[0].case == Case.PHYSICS:
        anchors = torch.linspace(0, 1, 5)
        prediction = logits.softmax(-1) @ anchors
        targets = torch.tensor([row.mean_score_rate for row in records])
        points = torch.tensor([row.points for row in records])
        return ((prediction - targets).square() * points).sum() / points.sum()
    return nn.functional.cross_entropy(logits, torch.tensor([row.choice for row in records]))


def predictions(model: DecisionModel, records: list[Record]) -> list[list[float]]:
    model.eval()
    result = []
    with torch.no_grad():
        for offset in range(0, len(records), 8):
            result.extend(model(*batch(records[offset : offset + 8])).softmax(-1).tolist())
    return result


def evaluate(records: list[Record], probabilities: list[list[float]]) -> dict:
    if records[0].case == Case.PHYSICS:
        means = [sum(value * index / 4 for index, value in enumerate(row)) for row in probabilities]
        errors = [abs(predicted - row.mean_score_rate) for predicted, row in zip(means, records, strict=True)]
        return {
            "mae": sum(errors) / len(errors),
            "points_weighted_mae": sum(error * row.points for error, row in zip(errors, records, strict=True))
            / sum(row.points for row in records),
            "within_5pp": sum(error <= 0.05 for error in errors) / len(errors),
            "within_10pp": sum(error <= 0.10 for error in errors) / len(errors),
        }
    return {
        "accuracy": sum(
            max(range(3), key=row.__getitem__) == record.choice
            for row, record in zip(probabilities, records, strict=True)
        )
        / len(records),
        "brier": sum(
            sum((p - float(index == record.choice)) ** 2 for index, p in enumerate(row))
            for row, record in zip(probabilities, records, strict=True)
        )
        / len(records),
    }


def train(runtime, job: dict, stop_after_step: int | None = None) -> str:
    """Train only the split's train IDs. Final-test IDs are never consumed here."""
    torch.set_num_threads(1)
    from .runtime import code_version

    if code_version() != job["code_version"]:
        raise DomainError("code_version_changed")
    request = job["request"]
    records = runtime.catalog.records(request["dataset_id"])
    split = runtime.catalog.load_split(request["split_id"], request["dataset_id"])
    by_id = {record.id: record for record in records}
    train_rows = [by_id[key] for key in split["assignments"]["train"]]
    validation = [by_id[key] for key in split["assignments"]["validation"]]
    if len(train_rows) < 8:
        raise DomainError("insufficient_training_group_size")
    folder = runtime.directory(job["id"])
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash(
        {
            "request": request,
            "dataset": split["dataset_sha256"],
            "code": job["code_version"],
            "model": MODEL_VERSION,
        }
    )
    model = DecisionModel(Case(job["case_name"]), request["recipe_id"], request["seed"])
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=0.005)
    checkpoint = folder / "checkpoint.pt"
    step, spent, curve = 0, 0.0, []
    if checkpoint.exists():
        manifest_path = folder / "checkpoint.sha256"
        if not manifest_path.is_file() or manifest_path.read_text().strip() != digest(checkpoint):
            raise DomainError("checkpoint_integrity_failed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise DomainError("checkpoint_version_mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        step, spent, curve = saved["step"], saved["spent_seconds"], saved["curve"]
    started = time.monotonic()
    state = "completed"
    initial_private = (
        psutil.Process().memory_info().private
        if hasattr(psutil.Process().memory_info(), "private")
        else psutil.Process().memory_info().rss
    )
    if not checkpoint.exists():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "step": step,
                "spent_seconds": spent,
                "curve": curve,
                "fingerprint": fingerprint,
            },
            checkpoint,
        )
        (folder / "checkpoint.sha256").write_text(digest(checkpoint), encoding="ascii")
    while step < request["max_steps"]:
        if runtime.job(job["id"])["stop_requested"] or (
            stop_after_step is not None and step >= stop_after_step
        ):
            state = "stopped"
            break
        if time.monotonic() - started + spent > request["max_seconds"]:
            raise DomainError("wall_time_budget_exceeded")
        info = psutil.Process().memory_info()
        private = info.private if hasattr(info, "private") else info.rss
        # This registered 17k-parameter CPU recipe reserves 2 GiB of host availability.
        # Target-model recipes need their own measured budget and remain disabled.
        if psutil.virtual_memory().available < 2 * 2**30 or private - initial_private > 512 * 2**20:
            raise DomainError("memory_budget_exceeded")
        offset = step * 8 % len(train_rows)
        selected = [train_rows[(offset + index) % len(train_rows)] for index in range(8)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(*batch(selected)), selected)
        if not torch.isfinite(loss):
            raise DomainError("nonfinite_training_loss")
        loss.backward()
        nn.utils.clip_grad_norm_((value for value in model.parameters() if value.requires_grad), 1.0)
        optimizer.step()
        step += 1
        curve.append(float(loss.detach()))
        temporary = folder / "checkpoint.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "step": step,
                "spent_seconds": spent + time.monotonic() - started,
                "curve": curve,
                "fingerprint": fingerprint,
            },
            temporary,
        )
        temporary.replace(checkpoint)
        (folder / "checkpoint.sha256").write_text(digest(checkpoint), encoding="ascii")
    probs = predictions(model, validation)
    grouped = defaultdict(list)
    for index, record in enumerate(validation):
        grouped[record.group].append(index)
    group_diagnostics = []
    for group, indices in grouped.items():
        group_diagnostics.append(
            {
                "group": group,
                "count": len(indices),
                "metrics": evaluate(
                    [validation[index] for index in indices], [probs[index] for index in indices]
                ),
            }
        )
    report = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "evidence_level": "synthetic_cpu_smoke",
        "steps": step,
        "train_curve": curve,
        "validation_count": len(validation),
        "independent_validation_groups": len(grouped),
        "validation": evaluate(validation, probs),
        "groups": group_diagnostics,
        "resource": {
            "cpu_threads": 1,
            "elapsed_seconds": spent + time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
        },
        "private_validation_ids": [record.id for record in validation],
        "private_probabilities": probs,
        "test_labels_consumed": False,
        "objective": "supervised_mean_mse"
        if job["case_name"] == Case.PHYSICS
        else "supervised_cross_entropy",
    }
    atomic_json(folder / "diagnostics.private.json", report)
    atomic_json(
        folder / "artifacts.json",
        {
            "checkpoint.pt": digest(checkpoint),
            "diagnostics.private.json": digest(folder / "diagnostics.private.json"),
        },
    )
    return state


def load_trained(runtime, run_id: str) -> DecisionModel:
    runtime.verify_artifacts(run_id)
    job = runtime.job(run_id)
    if job["state"] != "completed":
        raise DomainError("completed_model_required")
    model = DecisionModel(Case(job["case_name"]), job["request"]["recipe_id"], job["request"]["seed"])
    saved = torch.load(runtime.directory(run_id) / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    return model
