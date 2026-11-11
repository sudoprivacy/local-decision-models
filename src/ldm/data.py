import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import Case, DomainError, Record


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fixture_records(case: Case) -> list[Record]:
    records = []
    for group in range(6):
        for index in range(12):
            if case == Case.PHYSICS:
                difficulty = index % 3
                records.append(
                    Record(
                        id=f"synthetic-{group}-{index}",
                        case=case,
                        group=f"paper-{2020 + group}",
                        time_index=2020 + group,
                        context=f"Synthetic mechanics exercise. A {index + 1} kg block moves on a level surface.",
                        question=[
                            "State its acceleration with no horizontal force.",
                            "Find acceleration when a 6 N horizontal force is applied.",
                            "Explain motion when friction and a changing force act together.",
                        ][difficulty],
                        answer="Synthetic answer retained with the exercise.",
                        rubric="Synthetic scoring criteria; award the stated subquestion points.",
                        points=float(1 + index % 5),
                        mean_score_rate=[0.85, 0.6, 0.3][difficulty],
                        question_type="choice" if index % 2 else "written",
                    )
                )
            else:
                choice = index % 3
                records.append(
                    Record(
                        id=f"synthetic-{group}-{index}",
                        case=case,
                        group=f"episode-family-{group}",
                        time_index=group,
                        context=[
                            "Study room: reader present, low daylight.",
                            "Study room: guests meeting, normal daylight.",
                            "Study room: no request, comfortable current settings.",
                        ][choice],
                        question="Choose a recommendation: reading, meeting, or keep current settings.",
                        choice=choice,
                        question_type="scene",
                    )
                )
    return records


def physics_labels(path: Path) -> tuple[list[dict], dict]:
    """Read locally; public return contains only coverage, never row-level rates."""
    observed, seen, missing = [], set(), 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"year", "paper_id", "population", "question_id", "points", "actual_score_rate"}
        if not required.issubset(reader.fieldnames or []):
            raise DomainError("physics_columns_missing")
        for row in reader:
            try:
                year, points = int(row["year"]), float(row["points"])
                qid = row["question_id"]
                if not re.fullmatch(r"\d+(?:-\d+)*", qid) or not math.isfinite(points) or points <= 0:
                    raise ValueError
                key = (year, row["paper_id"], row["population"], qid)
                if key in seen:
                    raise DomainError("physics_duplicate_question")
                seen.add(key)
                if row["actual_score_rate"].strip() == "":
                    missing += 1
                    continue
                rate = float(row["actual_score_rate"])
                if not math.isfinite(rate) or not 0 <= rate <= 1:
                    raise ValueError
                observed.append({"key": key, "points": points, "rate": rate})
            except (TypeError, ValueError):
                raise DomainError("physics_invalid_value") from None
    groups = defaultdict(list)
    for key in seen:
        groups[key[:3]].append(key[3])
    for identifiers in groups.values():
        if any(b.startswith(a + "-") for a in identifiers for b in identifiers if a != b):
            raise DomainError("physics_parent_child_overlap")
    counts, coverage = Counter(), Counter()
    for row in observed:
        counts[row["key"][0]] += 1
        coverage[row["key"][0]] += row["points"]
    if any(points > 100 for points in coverage.values()):
        raise DomainError("physics_coverage_exceeds_full_paper")
    return observed, {
        "observed_rows": len(observed),
        "missing_labels": missing,
        "years": [
            {"year": year, "observed_rows": counts[year], "covered_points": coverage[year]}
            for year in sorted(counts)
        ],
        "missing_policy": "preserved_not_zero_filled",
    }


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        self.items = json.loads((root / "catalog.json").read_text(encoding="utf-8"))

    def item(self, dataset_id: str) -> dict:
        if dataset_id not in self.items:
            raise DomainError("dataset_not_registered")
        return self.items[dataset_id]

    def records(self, dataset_id: str) -> list[Record]:
        item = self.item(dataset_id)
        if item["kind"] == "physics_labels_only":
            raise DomainError("question_alignment_pending")
        path = self.root / item["relative_path"]
        if digest(path) != item["sha256"]:
            raise DomainError("dataset_version_changed")
        try:
            records = [
                Record.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except ValueError:
            raise DomainError("invalid_dataset_record") from None
        if len({row.id for row in records}) != len(records):
            raise DomainError("duplicate_record_id")
        if not records or any(row.case != item["case"] for row in records):
            raise DomainError("dataset_case_mismatch")
        return records

    def describe(self, dataset_id: str) -> dict:
        item = self.item(dataset_id)
        return {
            "dataset_id": dataset_id,
            "case": item["case"],
            "version": item["version"],
            "privacy": item["privacy"],
            "target": "cohort_mean_score_fraction" if item["case"] == Case.PHYSICS else "scene_choice",
            "allowed_views": ["overall", "by_group"],
            "minimum_group_count": 8,
            "fields": [
                "full_context",
                "question",
                "answer_if_supplied",
                "rubric_if_supplied",
                "group",
                "time_index",
            ],
            "available_recipes": ["cpu-head-v1", "cpu-lora-v1"] if item["privacy"] == "synthetic" else [],
        }

    def prepare(self, dataset_id: str) -> dict:
        item = self.item(dataset_id)
        if item["kind"] == "physics_labels_only":
            private_path = Path(item["private_path"])
            if digest(private_path) != item["sha256"]:
                raise DomainError("dataset_version_changed")
            _, coverage = physics_labels(private_path)
            return {
                "dataset_id": dataset_id,
                "version": item["version"],
                **coverage,
                "training_ready": False,
                "blockers": ["full_question_alignment_pending", "target_device_unverified"],
                "evidence_level": "real_labels_intake_only",
            }
        records = self.records(dataset_id)
        return {
            "dataset_id": dataset_id,
            "version": item["version"],
            "observed_rows": len(records),
            "groups": len({row.group for row in records}),
            "training_ready": True,
            "evidence_level": "synthetic_cpu_smoke",
            "blockers": [],
        }

    def split(self, dataset_id: str) -> dict:
        item, records = self.item(dataset_id), self.records(dataset_id)
        times = defaultdict(set)
        for row in records:
            times[row.group].add(row.time_index)
        if any(len(values) != 1 for values in times.values()):
            raise DomainError("group_crosses_time_boundary")
        units = sorted({row.time_index for row in records})
        if len(units) < 4:
            raise DomainError("insufficient_independent_time_groups")
        by_role = {"train": units[:-2], "validation": units[-2:-1], "test": units[-1:]}
        assignments = {
            role: [row.id for row in records if row.time_index in years] for role, years in by_role.items()
        }
        group_sets = [{row.group for row in records if row.id in assignments[role]} for role in by_role]
        if any(group_sets[a] & group_sets[b] for a in range(3) for b in range(a + 1, 3)):
            raise DomainError("split_group_leakage")
        value = {
            "dataset_id": dataset_id,
            "dataset_version": item["version"],
            "dataset_sha256": item["sha256"],
            "rule": "whole-group-time-v1",
            "assignments": assignments,
            "time_units": by_role,
            "test_role": "synthetic_holdout"
            if item["privacy"] == "synthetic"
            else "retrospective_not_independent_final",
        }
        split_id = canonical_hash(value)[:24]
        atomic_json(self.root / "splits" / (split_id + ".json"), value)
        return {
            "split_id": split_id,
            "dataset_version": item["version"],
            "counts": {role: len(ids) for role, ids in assignments.items()},
            "group_overlap": False,
            "time_leakage": False,
            "test_role": value["test_role"],
        }

    def load_split(self, split_id: str, dataset_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{24}", split_id):
            raise DomainError("invalid_split_id")
        path = self.root / "splits" / (split_id + ".json")
        if not path.is_file():
            raise DomainError("split_not_found")
        value = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(value)[:24] != split_id or value["dataset_id"] != dataset_id:
            raise DomainError("split_identity_mismatch")
        if value["dataset_sha256"] != self.item(dataset_id)["sha256"]:
            raise DomainError("split_dataset_version_mismatch")
        return value
