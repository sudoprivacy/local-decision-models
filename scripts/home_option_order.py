"""Deterministic, label-preserving option order augmentation for the home pilot.

The optional MS-SWIFT callback changes only lazy encoding. It leaves the dataset,
sampler and inference template intact. Runtime audits belong outside Git.
"""

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def option_orders(options):
    """Twelve distinct orders, independent of state and gold; cover every position.

    Keep the original and one-place rotation out of the training orders so the
    existing two evaluation orders still probe an unseen presentation.
    """
    if len(options) != 8 or len(set(options)) != 8:
        raise ValueError("eight_unique_options_required")
    rng = random.Random("home-order-r5:" + fingerprint(options))
    forbidden = {tuple(range(8)), tuple(range(1, 8)) + (0,)}
    orders = []
    for count in (8, 4):
        while True:
            base = list(range(8))
            rng.shuffle(base)
            rotations = [tuple(base[k:] + base[:k]) for k in range(count)]
            if not (set(rotations) & (forbidden | set(orders))):
                orders.extend(rotations)
                break
    return orders


def permute_prepared(row, epoch):
    if not isinstance(epoch, int) or not 0 <= epoch < 12:
        raise ValueError("epoch_must_be_in_0_to_11")
    if row.get("kinds") != ["choice"] or len(row.get("options", [])) != 1:
        raise ValueError("single_choice_question_required")
    options = row["options"][0]
    targets = row.get("target_probs")
    if targets is None or len(targets) != 1 or len(targets[0]) != 8:
        raise ValueError("eight_training_targets_required")
    order = option_orders(options)[epoch]
    result = copy.deepcopy(row)
    result["options"] = [[options[i] for i in order]]
    result["target_probs"] = [[targets[0][i] for i in order]]
    return result, order


def register_swift_callback():
    """Register through MS-SWIFT's external plugin / callback extension point."""
    import os

    from swift.callbacks import TrainerCallback, callbacks_map
    from swift.dataset import LazyLLMDataset

    class HomeOptionOrderCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.profile = os.environ.get("HOME_OPTION_ORDER_PROFILE", "standard")
            profiles = {"standard": 32, "minimal_pair": 2, "paired_guards": 8}
            if self.profile not in profiles:
                raise ValueError("unknown_training_exposure_profile")
            self.rows = profiles[self.profile]
            self.epochs = 384 // self.rows
            data = self.trainer.train_dataset
            if not isinstance(data, LazyLLMDataset) or len(data) != self.rows:
                raise ValueError("training_rows_do_not_match_exposure_profile")
            if (args.dataloader_num_workers != 0 or args.world_size != 1
                    or args.per_device_train_batch_size != 1
                    or args.gradient_accumulation_steps != 1 or args.max_steps != 384):
                raise ValueError("round5_training_contract_changed")
            if self.trainer.eval_dataset is not None or state.global_step != 0:
                raise ValueError("round5_requires_fresh_training_without_in_loop_evaluation")
            if kwargs["train_dataloader"].dataset is not data:
                raise ValueError("unexpected_dataloader_dataset_copy")
            self.audit_path = Path(os.environ["HOME_OPTION_ORDER_AUDIT"])
            if self.audit_path.exists():
                raise ValueError("augmentation_audit_already_exists")
            self.original_encode = data.encode_func
            self.epoch = None
            self.audit = []
            self.expected = Counter(fingerprint(data.dataset[i]) for i in range(len(data)))
            if len(self.expected) != self.rows:
                raise ValueError("duplicate_preprocessed_training_rows")
            # Fail closed on encoding errors; never replace a failed row with another.
            data.strict = True
            data.n_try_fetch = 1
            data.encode_func = self.encode

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch = int(state.global_step // self.rows)
            if state.global_step % self.rows or self.epoch >= self.epochs:
                raise ValueError("unexpected_epoch_boundary")

        def encode(self, row, **kwargs):
            if self.epoch is None:
                raise ValueError("encoding_outside_training_epoch")
            transformed, order = permute_prepared(row, self.epoch % 12)
            encoded = self.original_encode(transformed, **kwargs)
            if encoded["target_probs"] != transformed["target_probs"]:
                raise ValueError("encoded_target_mismatch")
            entry = {"epoch": self.epoch, "row_sha256": fingerprint(row),
                     "order": list(order), "token_count": len(encoded["input_ids"])}
            self.audit.append(entry)
            with self.audit_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(entry) + "\n")
            return encoded

        def on_epoch_end(self, args, state, control, **kwargs):
            seen = Counter(r["row_sha256"] for r in self.audit if r["epoch"] == self.epoch)
            if seen != self.expected or state.global_step != (self.epoch + 1) * self.rows:
                raise ValueError("epoch_exposure_contract_failed")
            self.epoch = None

        def on_train_end(self, args, state, control, **kwargs):
            if state.global_step != 384 or len(self.audit) != 384:
                raise ValueError("training_exposure_contract_failed")
            for key in self.expected:
                seen = [r for r in self.audit if r["row_sha256"] == key]
                order_counts = Counter(tuple(r["order"]) for r in seen)
                if (len(seen) != self.epochs or len(order_counts) != 12
                        or set(order_counts.values()) != {self.epochs // 12}):
                    raise ValueError("per_row_permutation_contract_failed")
            receipt = {"status": "passed", "steps": 384, "rows": self.rows,
                       "epochs": self.epochs, "profile": self.profile,
                       "exposures_per_row": self.epochs, "distinct_orders_per_row": 12,
                       "repeats_per_order_per_row": self.epochs // 12,
                       "training_encode_calls": len(self.audit),
                       "audit_sha256": hashlib.sha256(self.audit_path.read_bytes()).hexdigest(),
                       "max_tokens": max(r["token_count"] for r in self.audit),
                       "physical_commands_sent": 0}
            self.audit_path.with_name("augmentation-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    callbacks_map["home_option_order"] = HomeOptionOrderCallback
