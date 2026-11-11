import json

import pytest

from ldm.contracts import DomainError
from ldm.data import Catalog, atomic_json, digest, physics_labels


def test_group_and_time_split_is_immutable_and_disjoint(runtime):
    first = runtime.catalog.split("fixture-physics-v1")
    assert runtime.catalog.split("fixture-physics-v1") == first
    saved = runtime.catalog.load_split(first["split_id"], "fixture-physics-v1")
    train, val, test = map(set, saved["assignments"].values())
    assert not (train & val or train & test or val & test)
    records = runtime.catalog.records("fixture-physics-v1")
    groups = [{row.group for row in records if row.id in role} for role in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert max(saved["time_units"]["train"]) < min(saved["time_units"]["validation"])
    assert max(saved["time_units"]["validation"]) < min(saved["time_units"]["test"])


def test_changed_dataset_and_wrong_split_fail_closed(runtime):
    split = runtime.catalog.split("fixture-physics-v1")
    with pytest.raises(DomainError, match="split_identity_mismatch"):
        runtime.catalog.load_split(split["split_id"], "fixture-home-v1")
    path = runtime.root / runtime.catalog.item("fixture-physics-v1")["relative_path"]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(DomainError, match="dataset_version_changed"):
        runtime.catalog.records("fixture-physics-v1")


def test_related_group_cannot_cross_time(runtime):
    item = runtime.catalog.item("fixture-physics-v1")
    path = runtime.root / item["relative_path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["time_index"] = 2024
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    catalog = runtime.catalog.items
    catalog["fixture-physics-v1"]["sha256"] = digest(path)
    atomic_json(runtime.root / "catalog.json", catalog)
    with pytest.raises(DomainError, match="group_crosses_time_boundary"):
        Catalog(runtime.root).split("fixture-physics-v1")


def test_missing_real_label_is_not_zero_filled(tmp_path):
    path = tmp_path / "synthetic_intake.csv"
    path.write_text(
        "year,paper_id,population,question_id,points,actual_score_rate\n2020,p,c,1,2,0.4\n2020,p,c,2,3,\n",
        encoding="utf-8",
    )
    private_rows, public = physics_labels(path)
    assert len(private_rows) == public["observed_rows"] == 1
    assert public["missing_labels"] == 1
    assert "rate" not in json.dumps(public)


@pytest.mark.parametrize(
    "tail,code",
    [
        ("1-1,3,0.4", "physics_parent_child_overlap"),
        ("1,3,0.5", "physics_duplicate_question"),
        ("2,3,nan", "physics_invalid_value"),
    ],
)
def test_invalid_private_labels_are_rejected_without_values(tmp_path, tail, code):
    path = tmp_path / "synthetic.csv"
    path.write_text(
        "year,paper_id,population,question_id,points,actual_score_rate\n2020,p,c,1,2,0.4\n2020,p,c,"
        + tail
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DomainError, match=code):
        physics_labels(path)


def test_auth_and_validation_do_not_echo_submitted_secrets(client, auth):
    assert client.get("/v1/datasets/fixture-physics-v1").status_code == 401
    assert client.get("/v1/datasets/fixture-physics-v1", headers=auth["guest"]).status_code == 403
    response = client.post(
        "/v1/datasets/prepare",
        json={"dataset_id": "fixture-physics-v1", "sql": "SECRET_RAW_LABEL"},
        headers=auth["researcher"],
    )
    assert response.status_code == 422
    assert "SECRET_RAW_LABEL" not in response.text
    assert "sql" not in response.text
    assert client.get("/v1/datasets/../../credentials.json", headers=auth["researcher"]).status_code in {
        404,
        409,
    }


def test_catalog_and_prepare_return_no_local_paths(client, auth, runtime):
    response = client.get("/v1/datasets/fixture-physics-v1", headers=auth["researcher"])
    assert response.status_code == 200
    assert str(runtime.root) not in response.text
    assert "sha256" not in response.text
    prepared = client.post(
        "/v1/datasets/prepare", headers=auth["researcher"], json={"dataset_id": "fixture-physics-v1"}
    )
    assert prepared.json()["training_ready"] is True
