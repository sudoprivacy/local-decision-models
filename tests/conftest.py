import os

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1"
)

import pytest
from fastapi.testclient import TestClient

from ldm.api import create_app
from ldm.runtime import Runtime, bootstrap


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "private-runtime"
    bootstrap(root)
    return Runtime(root)


@pytest.fixture
def client(runtime):
    with TestClient(create_app(runtime.root), raise_server_exceptions=False) as value:
        yield value


@pytest.fixture
def auth(runtime):
    return {role: {"Authorization": "Bearer " + token} for role, token in runtime.credentials().items()}
