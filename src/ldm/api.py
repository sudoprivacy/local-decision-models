import secrets
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import __version__
from .contracts import (
    ComparisonRequest,
    DatasetRequest,
    Diagnostics,
    DomainError,
    HomeRequest,
    RunRequest,
    ShadowApproval,
    SplitRequest,
)
from .data import Catalog
from .diagnostics import compare, diagnostics
from .home import preview
from .runtime import Runtime


def create_app(root: Path) -> FastAPI:
    runtime = Runtime(root)
    credentials = runtime.credentials()
    if set(credentials) != {"owner", "researcher", "guest"} or len(set(credentials.values())) != 3:
        raise DomainError("invalid_service_credentials")
    app = FastAPI(title="Local decision models", version=__version__)
    auth = HTTPBearer(auto_error=False)

    def identity(value: Annotated[HTTPAuthorizationCredentials | None, Depends(auth)]):
        if value is None:
            raise DomainError("authentication_required")
        for role, token in credentials.items():
            if len(token) < 24:
                raise DomainError("invalid_service_credentials")
            if secrets.compare_digest(value.credentials, token):
                return role
        raise DomainError("authentication_required")

    def researcher(role=Depends(identity)):
        if role not in {"owner", "researcher"}:
            raise DomainError("permission_denied")
        return role

    def owner(role=Depends(identity)):
        if role != "owner":
            raise DomainError("permission_denied")
        return role

    @app.exception_handler(DomainError)
    async def domain_error(_request, error):
        code = (
            401
            if error.code == "authentication_required"
            else 403
            if error.code == "permission_denied"
            else 409
        )
        return JSONResponse(status_code=code, content={"error_code": error.code})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        return JSONResponse(status_code=422, content={"error_code": "invalid_request"})

    @app.exception_handler(Exception)
    async def internal_error(_request, _error):
        return JSONResponse(status_code=500, content={"error_code": "internal_error"})

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "mode": "local_reference"}

    @app.get("/v1/datasets/{dataset_id}", dependencies=[Depends(researcher)])
    def describe_dataset(dataset_id: str):
        return Catalog(root).describe(dataset_id)

    @app.post("/v1/datasets/prepare", dependencies=[Depends(researcher)])
    def prepare_dataset(request: DatasetRequest):
        return Catalog(root).prepare(request.dataset_id)

    @app.post("/v1/splits", dependencies=[Depends(researcher)])
    def create_split(request: SplitRequest):
        return Catalog(root).split(request.dataset_id)

    @app.post("/v1/experiments", status_code=202, dependencies=[Depends(researcher)])
    def run_experiment(request: RunRequest):
        return runtime.enqueue(request)

    @app.get(
        "/v1/experiments/{run_id}/diagnostics", response_model=Diagnostics, dependencies=[Depends(researcher)]
    )
    def get_diagnostics(run_id: str, view: Annotated[Literal["overall", "by_group"], Query()] = "overall"):
        return diagnostics(runtime, run_id, view)

    @app.post("/v1/experiments/{run_id}/stop", dependencies=[Depends(researcher)])
    def stop_experiment(run_id: str):
        return runtime.stop(run_id)

    @app.post("/v1/experiments/{run_id}/resume", dependencies=[Depends(owner)])
    def resume_experiment(run_id: str):
        return runtime.resume(run_id)

    @app.post("/v1/experiments/compare", dependencies=[Depends(researcher)])
    def compare_runs(request: ComparisonRequest):
        return compare(runtime, request.run_ids)

    @app.post("/v1/experiments/{run_id}/shadow", dependencies=[Depends(owner)])
    def approve_shadow(run_id: str, request: ShadowApproval):
        return runtime.shadow(run_id, request.note)

    @app.post("/v1/experiments/{run_id}/rollback", dependencies=[Depends(owner)])
    def rollback(run_id: str, request: ShadowApproval):
        return runtime.shadow(run_id, request.note, rollback=True)

    @app.post("/v1/home/preview")
    def home_preview(request: HomeRequest, role=Depends(identity)):
        return preview(runtime, request, role)

    return app
