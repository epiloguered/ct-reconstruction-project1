"""Small numerical checks for the data path and ART update."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from run_baselines import cyclic_art, load_matrix, read_sparse_csv


class BaselineTests(unittest.TestCase):
    def test_published_checkout_loads_npz_without_original_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "Data" / "SheppLogan64"
            data.mkdir(parents=True)
            expected = sparse.csr_matrix(([2.0, 3.0], ([0, 2], [7, 4095])), shape=(3, 4096))
            sparse.save_npz(data / "A_csr.npz", expected)
            with patch("run_baselines.ROOT", root):
                actual = load_matrix("SheppLogan64", root / "cache")
            self.assertFalse((data / "A.csv").exists())
            np.testing.assert_array_equal(actual.indptr, expected.indptr)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            np.testing.assert_array_equal(actual.data, expected.data)

    def test_sparse_csv_keeps_zero_rows_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.csv"
            path.write_text("0,2,0,0\n0,0,0,0\n1,0,3,0\n", encoding="utf-8")
            matrix = read_sparse_csv(path, 4)
            np.testing.assert_array_equal(matrix.toarray(), [[0, 2, 0, 0], [0, 0, 0, 0], [1, 0, 3, 0]])

    def test_art_recovers_nonnegative_identity_and_skips_zero_row(self):
        matrix = sparse.vstack([sparse.eye(64, format="csr"), sparse.csr_matrix((1, 64))], format="csr")
        truth = np.arange(64, dtype=float).reshape(8, 8)
        x_true = truth.reshape(-1, order="F")
        b = np.r_[x_true, 7.0]  # Inconsistent zero row must be skipped safely.
        final_x, fixed_x, best_x, history, best_epoch = cyclic_art(
            matrix, b, truth, epochs=2, relaxation=1.0, fixed_epoch=1
        )
        np.testing.assert_allclose(final_x, x_true, atol=1e-12)
        np.testing.assert_allclose(fixed_x, x_true, atol=1e-12)
        np.testing.assert_allclose(best_x, x_true, atol=1e-12)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(len(history), 2)
        np.testing.assert_allclose(lsqr(matrix, b)[0], x_true, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
