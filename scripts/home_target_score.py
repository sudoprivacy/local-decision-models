"""Score the private synthetic bundle with a freshly loaded target model/adapter."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_engine(model, adapter):
    import torch

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("target_gpu_required")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 25 * 2**30 / total))
    import home_swift_plugin
    from swift.infer_engine import TransformersEngine
    from transformers import BitsAndBytesConfig

    engine = TransformersEngine(
        str(model),
        model_type=home_swift_plugin.MODEL_TYPE,
        task_type="decision",
        adapters=[str(adapter)] if adapter else None,
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        device_map={"": 0},
        max_batch_size=1,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["model.visual", "model.visual.merger", "lm_head"],
        ),
    )
    engine.template.max_length = 2048
    return engine


def score_options(engine, state, question):
    import home_swift_plugin
    import torch
    from swift.infer_engine import RequestConfig

    request = home_swift_plugin.home_decision_request(state, question)
    torch.cuda.synchronize()
    started = time.perf_counter()
    response = engine.infer([request], RequestConfig(stream=False), use_tqdm=False)[0]
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    if isinstance(response, Exception):
        raise response
    content = response.choices[0].message.content
    if isinstance(content, str):
        content = json.loads(content)
    return {"probabilities": content[0]["probs"], "latency_ms": elapsed}


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "bundle.json").read_text())
    if digest(args.bundle / "records.json") != manifest["records_sha256"]:
        raise RuntimeError("bundle_integrity_failed")
    rows = json.loads((args.bundle / "records.json").read_text())
    if args.limit:
        rows = rows[: args.limit]
    engine = load_engine(args.model, args.adapter)
    results = []
    for row in rows:
        state, question = row["model_input"]["state"], row["model_input"]["questions"][0]
        results.append({"id": row["id"], **score_options(engine, state, question)})
        print(json.dumps({"scored": len(results), "total": len(rows)}), flush=True)
    adapter_files = {}
    if args.adapter:
        for name in ("adapter_model.safetensors", "scoring_head.safetensors", "scoring_head.json"):
            adapter_files[name] = digest(args.adapter / name)
    output = {
        "schema_version": 1,
        "decision_source": "ms_swift_qwen3_5_9b_single_forward",
        "model_id": "Qwen/Qwen3.5-9B-Base",
        "model_revision": "modelscope:f3507de96f70f0056e2d441fe4cc6e9ded117382",
        "backbone_precision": "bitsandbytes_nf4_double_quant_bf16_compute",
        "bundle_sha256": digest(args.bundle / "bundle.json"),
        "checkpoint_sha256": adapter_files.get("adapter_model.safetensors"),
        "adapter_artifacts": adapter_files,
        "gpu": torch.cuda.get_device_name(0),
        "loaded_parameter_count": sum(p.numel() for p in engine.model.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "calibration_established": False,
        "rows": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
