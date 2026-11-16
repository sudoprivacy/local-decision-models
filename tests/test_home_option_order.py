import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from home_option_order import inference_row_key, option_orders, permute_prepared


class OptionOrderTests(unittest.TestCase):
    def test_all_actions_keep_their_labels_and_cover_every_position(self):
        options = ["on", "off", "reading", "manual", "refresh", "unsupported", "clarify", "keep"]
        for gold in range(8):
            row = {
                "messages": [{"role": "user", "content": "unchanged state"}],
                "kinds": ["choice"],
                "questions": ["what next?"],
                "options": [options],
                "target_probs": [[float(i == gold) for i in range(8)]],
            }
            before = copy.deepcopy(row)
            positions, orders = [], set()
            for epoch in range(12):
                changed, order = permute_prepared(row, epoch)
                index = changed["target_probs"][0].index(1.0)
                self.assertEqual(changed["options"][0][index], options[gold])
                self.assertEqual(changed["messages"], row["messages"])
                self.assertEqual(changed["questions"], row["questions"])
                positions.append(index)
                orders.add(order)
            self.assertEqual(set(positions), set(range(8)))
            self.assertEqual(len(orders), 12)
            self.assertNotIn(tuple(range(8)), orders)
            self.assertNotIn((1, 2, 3, 4, 5, 6, 7, 0), orders)
            self.assertEqual(row, before)

    def test_soft_targets_move_with_their_options_without_changing_rng(self):
        options = [str(i) for i in range(8)]
        row = {
            "kinds": ["choice"],
            "options": [options],
            "target_probs": [[0.05, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]],
        }
        rng_state = random.getstate()
        changed, _ = permute_prepared(row, 3)
        self.assertEqual(
            dict(zip(*[row[k][0] for k in ("options", "target_probs")])),
            dict(zip(*[changed[k][0] for k in ("options", "target_probs")])),
        )
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(option_orders(options), option_orders(options))

    def test_rejects_missing_targets_and_bad_epoch(self):
        row = {"kinds": ["choice"], "options": [[str(i) for i in range(8)]]}
        with self.assertRaises(ValueError):
            permute_prepared(row, 0)
        with self.assertRaises(ValueError):
            permute_prepared(row, 12)

    def test_historical_shared_orders_are_unchanged(self):
        self.assertEqual(
            option_orders([str(i) for i in range(8)]),
            [
                (3, 1, 4, 7, 5, 2, 0, 6),
                (1, 4, 7, 5, 2, 0, 6, 3),
                (4, 7, 5, 2, 0, 6, 3, 1),
                (7, 5, 2, 0, 6, 3, 1, 4),
                (5, 2, 0, 6, 3, 1, 4, 7),
                (2, 0, 6, 3, 1, 4, 7, 5),
                (0, 6, 3, 1, 4, 7, 5, 2),
                (6, 3, 1, 4, 7, 5, 2, 0),
                (6, 3, 4, 2, 1, 5, 0, 7),
                (3, 4, 2, 1, 5, 0, 7, 6),
                (4, 2, 1, 5, 0, 7, 6, 3),
                (2, 1, 5, 0, 7, 6, 3, 4),
            ],
        )

    def test_row_conditioned_orders_ignore_labels_and_metadata(self):
        row = {
            "messages": [{"role": "user", "content": "state A"}],
            "kinds": ["choice"],
            "questions": ["what next?"],
            "options": [[str(i) for i in range(8)]],
            "target_probs": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        }
        before = copy.deepcopy(row)
        rng_before = random.getstate()
        layouts = [permute_prepared(row, e, mode="row_conditioned")[1] for e in range(12)]
        for gold in range(8):
            altered = copy.deepcopy(row)
            altered.update(
                target_probs=[[float(i == gold) for i in range(8)]],
                dataset="different/path",
                id=f"label-{gold}",
                annotation={"gold": gold},
            )
            self.assertEqual(inference_row_key(altered), inference_row_key(row))
            positions = []
            for epoch in range(12):
                changed, order = permute_prepared(altered, epoch, mode="row_conditioned")
                self.assertEqual(order, layouts[epoch])
                position = changed["target_probs"][0].index(1.0)
                self.assertEqual(changed["options"][0][position], str(gold))
                self.assertEqual(changed["messages"], row["messages"])
                self.assertEqual(changed["questions"], row["questions"])
                positions.append(position)
            self.assertEqual(set(positions), set(range(8)))
        self.assertEqual(len(set(layouts)), 12)
        self.assertNotIn(tuple(range(8)), layouts)
        self.assertNotIn((1, 2, 3, 4, 5, 6, 7, 0), layouts)
        soft = copy.deepcopy(row)
        soft["target_probs"] = [[0.05, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]]
        for epoch in range(12):
            changed, order = permute_prepared(soft, epoch, mode="row_conditioned")
            self.assertEqual(order, layouts[epoch])
            self.assertEqual(
                dict(zip(soft["options"][0], soft["target_probs"][0])),
                dict(zip(changed["options"][0], changed["target_probs"][0])),
            )
        other = copy.deepcopy(row)
        other["messages"][0]["content"] = "state B"
        self.assertNotEqual(
            layouts, [permute_prepared(other, e, mode="row_conditioned")[1] for e in range(12)]
        )
        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(row, before)

    def test_row_conditioned_requires_input_and_rejects_unknown_modes(self):
        row = {
            "kinds": ["choice"],
            "options": [[str(i) for i in range(8)]],
            "target_probs": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        }
        for mode, reason in [
            ("row_conditioned", "inference_fields_required"),
            ("typo", "unknown_option_order_mode"),
        ]:
            with self.assertRaisesRegex(ValueError, reason):
                permute_prepared(row, 0, mode=mode)


if __name__ == "__main__":
    unittest.main()
