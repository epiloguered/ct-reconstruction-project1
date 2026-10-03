"""Angle geometry and saved classical experiment reproducibility checks."""
from __future__ import annotations

import json
import unittest

import numpy as np
from scipy import sparse

from run_baselines import DATASETS, ROOT, image_metrics, load_matrix
from run_sparse_views import ANGLES, DEFAULT_OUTPUT, METHODS, select_angles


class AngleSelectionTests(unittest.TestCase):
    def test_complete_uniform_angle_blocks_and_observation_alignment(self):
        n_lines = 3
        matrix = sparse.csr_matrix(np.arange(90*n_lines*4).reshape(90*n_lines, 4))
        b = np.arange(matrix.shape[0]) + 1000
        # Entirely zero detector row must stay in its selected block.
        matrix.data[matrix.indptr[18]:matrix.indptr[19]] = 0
        matrix.eliminate_zeros()
        sub, obs, angle_ids, row_ids = select_angles(matrix, b, 15)
        np.testing.assert_array_equal(angle_ids, np.arange(0, 90, 6))
        expected_rows = np.concatenate([np.arange(j*3, j*3+3) for j in angle_ids])
        np.testing.assert_array_equal(row_ids, expected_rows)
        np.testing.assert_array_equal(obs, b[expected_rows])
        np.testing.assert_array_equal(sub.toarray(), matrix.toarray()[expected_rows])
        self.assertEqual(sub.shape, (45, 4))
        self.assertEqual(sub[3].nnz, 0)

    def test_full_view_is_identical_and_invalid_inputs_rejected(self):
        matrix = sparse.eye(180, format="csr")
        b = np.arange(180)
        sub, obs, angle_ids, row_ids = select_angles(matrix, b, 90)
        self.assertEqual((sub != matrix).nnz, 0)
        np.testing.assert_array_equal(obs, b)
        np.testing.assert_array_equal(row_ids, np.arange(180))
        for count in (0, 91, 1.5):
            with self.assertRaises(ValueError):
                select_angles(matrix, b, count)
        with self.assertRaises(ValueError):
            select_angles(sparse.eye(181, format="csr"), np.zeros(181), 15)
        with self.assertRaises(ValueError):
            select_angles(matrix, b[:-1], 15)

    def test_column_major_truth_roundtrip(self):
        truth = np.arange(64, dtype=float).reshape((8, 8))
        matrix = sparse.eye(64, format="csr")
        x = truth.reshape(-1, order="F")
        scores = image_metrics(x, truth, matrix, x)
        self.assertEqual(scores["nrmse"], 0)
        self.assertEqual(scores["relative_residual"], 0)
        self.assertEqual(scores["ssim"], 1)


class SavedExperimentTests(unittest.TestCase):
    def test_all_saved_selected_metrics_and_oracle_choices(self):
        records = sorted((DEFAULT_OUTPUT / "selected_records").glob("*.json"))
        self.assertEqual(len(records), len(DATASETS)*len(ANGLES)*4*len(METHODS))
        matrices = {name: load_matrix(name, ROOT / "outputs" / "cache") for name in DATASETS}
        truths = {name: np.loadtxt(ROOT / "Data" / name / "gt.csv", delimiter=",") for name in DATASETS}
        candidates = [json.loads(p.read_text(encoding="utf-8"))["metrics"]
                      for p in (DEFAULT_OUTPUT / "candidate_records").glob("*.json")]
        self.assertEqual(len(candidates), 2*4*4*(1+1+8+9+8))
        for path in records:
            row = json.loads(path.read_text(encoding="utf-8"))
            name, snr, angles, method = row["dataset"], row["snr_label_db"], row["angle_count"], row["method"]
            b = np.loadtxt(ROOT / "Data" / name / f"b_hat_{snr}db.csv", delimiter=",").reshape(-1)
            sub, obs, _, _ = select_angles(matrices[name], b, angles)
            x = np.load(DEFAULT_OUTPUT / row["selected_array"])
            candidate_x = np.load(DEFAULT_OUTPUT / row["candidate_array"])
            np.testing.assert_array_equal(x, candidate_x)
            self.assertTrue(np.isfinite(x).all())
            scores = image_metrics(x, truths[name], sub, obs)
            for key, score in scores.items():
                self.assertAlmostEqual(score, row[key], places=11, msg=f"{path.name}: {key}")
            grid = [r for r in candidates if (r["dataset"], r["angle_count"], r["snr_label_db"], r["method"])
                    == (name, angles, snr, method)]
            self.assertEqual(len(grid), row["grid_size"])
            self.assertEqual(row["nrmse"], min(r["nrmse"] for r in grid))
            if method == "LS_L2":
                self.assertAlmostEqual(row["lambda_raw"], row["lambda_normalized"]*sub.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
