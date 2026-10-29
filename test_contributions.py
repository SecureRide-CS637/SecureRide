#!/usr/bin/env python3
"""Tests for contributions 1 (class imbalance) and 2 (temporal smoothing). Run: python3 test_contributions.py
Uses only numpy/pandas/scikit-learn (no PyTorch needed)."""
import unittest
import numpy as np
import secureride_data as sd
from secureride_model import resolve_class_weight, make_predictor
from dualmode import majority_vote, MajorityVote, dual_from_slots
import imbalance as imb
from smoothing import zones, Acc


class TestSmoothing(unittest.TestCase):
    def test_n1_is_identity(self):
        p = np.array([0, 1, 0, 1, 1, 0], np.int8)
        self.assertTrue((majority_vote(p, 1) == p).all())

    def test_removes_isolated_flicker(self):
        p = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1], np.int8)
        s = majority_vote(p, 3)
        self.assertEqual(s[2], 0)                     # single 1 in a legal stretch is removed
        self.assertEqual(s[10], 1)                    # single 0 inside an illegal stretch does not break it

    def test_causal_and_streaming_equal(self):
        rng = np.random.default_rng(0)
        p = (rng.random(300) < 0.4).astype(np.int8)
        for n in (3, 5, 9):
            mv = MajorityVote(n)
            stream = np.array([mv.push(v) for v in p], np.int8)
            self.assertTrue((stream == majority_vote(p, n)).all())
            q = p.copy(); q[200:] = 1 - q[200:]       # changing the FUTURE must not change the past
            self.assertTrue((majority_vote(q, n)[:200] == majority_vote(p, n)[:200]).all())

    def test_delay_is_half_window(self):
        p = np.r_[np.zeros(20), np.ones(20)].astype(np.int8)
        for n in (3, 5, 7):
            first = int(np.argmax(majority_vote(p, n)))
            self.assertEqual(first - 20, (n + 1) // 2 - 1)

    def test_dual_with_smoothing_runs(self):
        pq = np.zeros(100, np.int8); pq[30] = 1       # lone false alarm
        pa = np.zeros(100, np.int8)
        out, sw = dual_from_slots(pq, pa, 5, 20)
        self.assertGreater(sw, 0)
        out_s, sw_s = dual_from_slots(majority_vote(pq, 3), pa, 5, 20)
        self.assertEqual(sw_s, 0); self.assertEqual(int(out_s.sum()), 0)

    def test_zones_and_false_alarm_counting(self):
        y = np.r_[np.zeros(30), np.ones(30), np.zeros(30)].astype(np.int8)
        legal, ill = zones(y, 10)
        self.assertTrue(legal[9] and ill[39] and not legal[35] and not ill[35])
        out = np.zeros(90, np.int8); out[15] = 1; out[16] = 1; out[70] = 1
        a = Acc(); a.add(out, legal, ill)
        self.assertEqual(a.fa_events, 2); self.assertEqual(a.fa_slots, 3)


class TestImbalance(unittest.TestCase):
    def test_balanced_weights(self):
        w = resolve_class_weight("balanced", np.array([0] * 95 + [1] * 5))
        self.assertAlmostEqual(w[1] / w[0], 19.0, places=6)
        self.assertIsNone(resolve_class_weight(None, np.array([0, 1])))
        self.assertEqual(list(resolve_class_weight((1, 4), np.array([0, 1]))), [1, 4])

    def test_prevalence_hit_and_no_leak(self):
        for prev in (0.5, 0.05, 0.01):
            for split in ("train", "val", "test"):
                X, y = imb.make_set("data", "sidewalk", 10, split, prev, 8000, seed=0)
                self.assertAlmostEqual(y.mean(), prev, delta=0.005 + 0.1 * prev)
                self.assertEqual(X.shape[1:], (10, 3))
        z = sd._npz("data")
        for split in ("train", "val", "test"):
            st, _ = imb.all_windows("data", "sidewalk", 10, split)
            self.assertTrue((z["split"][st] == split).all())

    def test_labels_follow_all_true_rule(self):
        z = sd._npz("data")
        st, y = imb.all_windows("data", "speeding", 5, "test")
        i = np.arange(0, len(st), 997)[:200]
        for k in i:
            w = z["speeding"][st[k]:st[k] + 5]
            self.assertEqual(int(w.all()), int(y[k]))

    def test_weighted_sklearn_raises_recall(self):
        Xtr, ytr = imb.make_set("data", "sidewalk", 10, "train", 0.05, 6000, 0)
        Xte, yte = imb.make_set("data", "sidewalk", 10, "test", 0.05, 6000, 0)
        rec = {}
        for cw in (None, "balanced"):
            p = make_predictor("sklearn", 10, 1, class_weight=cw); p.fit(Xtr, ytr)
            rec[cw] = ((p.predict(Xte) == 1) & (yte == 1)).sum() / yte.sum()
        self.assertGreater(rec["balanced"], rec[None])


if __name__ == "__main__":
    unittest.main(verbosity=2)
