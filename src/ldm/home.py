import math

from .contracts import Case, DomainError, HomeIntent, HomeRequest, Record


def policy(request: HomeRequest, role: str, probabilities: list[float] | None) -> dict:
    """Deterministic simulator gate; model outputs grant no authority."""
    base = {"gateway": "simulator", "physical_commands_sent": 0, "manual_path_available": True}
    if role not in {"owner", "guest"}:
        return {**base, "status": "denied", "reason": "home_permission_required"}
    if role == "guest" and request.space != "guest_room":
        return {**base, "status": "denied", "reason": "space_not_authorized"}
    if request.intent == HomeIntent.ENTRY:
        return {**base, "status": "denied", "reason": "entry_protocol_requires_separate_acceptance"}
    if not request.model_online:
        return {**base, "status": "manual_fallback", "reason": "model_offline"}
    if request.intent == HomeIntent.AMBIGUOUS:
        return {**base, "status": "clarify", "reason": "request_ambiguous"}
    if probabilities is None:
        return {**base, "status": "clarify", "reason": "approved_model_unavailable"}
    if (
        len(probabilities) != 3
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise DomainError("invalid_model_decision")
    if max(probabilities) < 0.5 or probabilities[0] != max(probabilities):
        return {**base, "status": "clarify", "reason": "model_uncertain_or_intent_mismatch"}
    if role != "owner" or not request.human_confirmed:
        return {**base, "status": "confirmation_required", "reason": "owner_confirmation_required"}
    return {**base, "status": "simulated", "scene": "reading", "space": request.space}


def preview(runtime, request: HomeRequest, role: str) -> dict:
    early = policy(request, role, None)
    if early["reason"] != "approved_model_unavailable":
        return early
    run_id = runtime.active_shadow(Case.HOME)
    if run_id is None:
        return early
    from .training import load_trained, predictions

    model = load_trained(runtime, run_id)
    record = Record(
        id="local-preview",
        case=Case.HOME,
        group="preview",
        time_index=0,
        context="Study room: reader present, low daylight.",
        question="Choose a recommendation: reading, meeting, or keep current settings.",
        choice=0,
        question_type="scene",
    )
    return {
        **policy(request, role, predictions(model, [record])[0]),
        "model_run_id": run_id,
        "evidence_level": "synthetic_cpu_smoke",
    }
