"""Qwen3.5 loader + upstream ms-swift decision head/template/trainer.

Used only by the standalone GPU pilot, outside the CPU research API.
"""

import json
import os
from pathlib import Path

from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import JevPreprocessor
from swift.model import ModelMeta, register_model
from swift.model.decision_head import JevVerbalizerHead, maybe_load_trained_scoring_head
from swift.model.model_arch import ModelArch
from swift.model.models.decision import ScoringModelLoader
from swift.model.models.qwen import Qwen3_5Loader

MODEL_TYPE = "home_qwen3_5_decision"


def serialize_home_state(state):
    """Preserve nested device/profile state identically in training and scoring."""
    if not isinstance(state, dict):
        raise ValueError("home_decision_state_must_be_object")
    return json.dumps(state, ensure_ascii=False, allow_nan=False)


def home_decision_request(state, question):
    return {
        "messages": [{"role": "user", "content": serialize_home_state(state)}],
        "kinds": ["choice"],
        "questions": [question["question"]],
        "options": [question["options"]],
    }


class HomeDecisionPreprocessor(JevPreprocessor):
    # The generic JEV preprocessor keeps only top-level string values from dicts.
    # Home decisions also require nested device, freshness and preference fields.
    @staticmethod
    def _state_to_text(state):
        return serialize_home_state(state)


class HomeQwenDecisionLoader(ScoringModelLoader, Qwen3_5Loader):
    """Use the official hybrid/VL loader; train a new generic decision readout."""

    def get_model(self, model_dir, config, processor, model_kwargs):
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, 25 * 2**30 / total))
        if config.model_type != "qwen3_5":
            raise ValueError("home_pilot_requires_qwen3_5_config")
        config.text_config.use_cache = False
        self.home_tokenizer = self._get_tokenizer(processor)
        model = super().get_model(model_dir, config, processor, model_kwargs)
        maybe_load_trained_scoring_head(model, model_dir)
        return model

    def build_head(self, model, model_dir, config):
        return JevVerbalizerHead.from_tokenizer(self.home_tokenizer, config=config)


register_model(
    ModelMeta(
        MODEL_TYPE,
        [],
        HomeQwenDecisionLoader,
        template="generic_decision",
        model_arch=ModelArch.qwen2_vl,
        task_type="decision",
        is_multimodal=True,
        architectures=[],
        tags=["vision", "decision"],
    )
)

if os.environ.get("HOME_PILOT_BUNDLE"):
    bundle = Path(os.environ["HOME_PILOT_BUNDLE"])
    for split in ("train", "validation", "test"):
        register_dataset(
            DatasetMeta(
                dataset_name=f"home_pilot_{split}",
                dataset_path=str(bundle / f"{split}.jsonl"),
                preprocess_func=HomeDecisionPreprocessor(),
            ),
            exist_ok=True,
        )
