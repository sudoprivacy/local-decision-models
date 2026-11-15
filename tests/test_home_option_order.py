import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from home_option_order import option_orders, permute_prepared


class OptionOrderTests(unittest.TestCase):
    def test_all_actions_keep_their_labels_and_cover_every_position(self):
        options = ["on", "off", "reading", "manual", "refresh", "unsupported", "clarify", "keep"]
        for gold in range(8):
            row = {"messages": [{"role": "user", "content": "unchanged state"}],
                   "kinds": ["choice"], "questions": ["what next?"], "options": [options],
                   "target_probs": [[float(i == gold) for i in range(8)]]}
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
        row = {"kinds": ["choice"], "options": [options],
               "target_probs": [[0.05, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]]}
        rng_state = random.getstate()
        changed, _ = permute_prepared(row, 3)
        self.assertEqual(dict(zip(*[row[k][0] for k in ("options", "target_probs")])),
                         dict(zip(*[changed[k][0] for k in ("options", "target_probs")])) )
        self.assertEqual(random.getstate(), rng_state)
        self.assertEqual(option_orders(options), option_orders(options))

    def test_rejects_missing_targets_and_bad_epoch(self):
        row = {"kinds": ["choice"], "options": [[str(i) for i in range(8)]]}
        with self.assertRaises(ValueError):
            permute_prepared(row, 0)
        with self.assertRaises(ValueError):
            permute_prepared(row, 12)


if __name__ == "__main__":
    unittest.main()
