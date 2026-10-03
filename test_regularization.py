"""Numerical checks of the new proximal and regularized ART steps."""

import unittest

import numpy as np
from scipy import sparse

from run_regularization import art_with_prior, fista, proximal, spectral_lipschitz


class RegularizationTests(unittest.TestCase):
    def test_l1_fista_matches_identity_soft_threshold(self):
        matrix = sparse.eye(64, format="csr")
        b = np.linspace(-2, 2, 64)
        lipschitz = spectral_lipschitz(matrix)
        x, history = fista(matrix, b, "L1", lam=0.3, lipschitz=lipschitz,
                           shape=(8, 8), iterations=100)
        expected = np.sign(b) * np.maximum(np.abs(b) - 0.3, 0)
        np.testing.assert_allclose(x, expected, atol=1e-8)
        self.assertLess(history[-1]["objective"], history[0]["objective"])

    def test_tv_prox_preserves_constant_image(self):
        constant = np.full(64, 0.7)
        np.testing.assert_allclose(proximal(constant, "TV", 0.2, (8, 8)), constant, atol=1e-12)

    def test_art_l2_shrink_and_zero_row(self):
        matrix = sparse.vstack([sparse.eye(64, format="csr"), sparse.csr_matrix((1, 64))], format="csr")
        b = np.r_[np.ones(64), 100.0]
        x = art_with_prior(matrix, b, "L2", strength=1.0, shape=(8, 8), epochs=1)
        np.testing.assert_allclose(x, 0.5 * np.ones(64), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
