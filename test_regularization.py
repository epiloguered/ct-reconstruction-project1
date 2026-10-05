# 正则化数值测试：用解析解和简单不变量检查近端处理与 ART 先验。
#
"""Numerical checks of the new proximal and regularized ART steps."""

import unittest

import numpy as np
from scipy import sparse

from run_regularization import art_with_prior, fista, proximal, spectral_lipschitz


# 分别覆盖 L1 解析解、TV 常数图像不变量及 L2 缩小／零行边界。
class RegularizationTests(unittest.TestCase):
    # 单位矩阵的 L1 解是软阈值，可直接检查 FISTA 是否求到正确结果。
    def test_l1_fista_matches_identity_soft_threshold(self):
        matrix = sparse.eye(64, format="csr")
        b = np.linspace(-2, 2, 64)
        lipschitz = spectral_lipschitz(matrix)
        x, history = fista(matrix, b, "L1", lam=0.3, lipschitz=lipschitz,
                           shape=(8, 8), iterations=100)
        expected = np.sign(b) * np.maximum(np.abs(b) - 0.3, 0)
        np.testing.assert_allclose(x, expected, atol=1e-8)
        self.assertLess(history[-1]["objective"], history[0]["objective"])

    # 常数图像 TV 为零，TV 近端处理不应改变它。
    def test_tv_prox_preserves_constant_image(self):
        constant = np.full(64, 0.7)
        np.testing.assert_allclose(proximal(constant, "TV", 0.2, (8, 8)), constant, atol=1e-12)

    # 检查 ART 每轮结束后的 L2 缩小比例，并确认安全跳过零行。
    def test_art_l2_shrink_and_zero_row(self):
        matrix = sparse.vstack([sparse.eye(64, format="csr"), sparse.csr_matrix((1, 64))], format="csr")
        b = np.r_[np.ones(64), 100.0]
        x = art_with_prior(matrix, b, "L2", strength=1.0, shape=(8, 8), epochs=1)
        np.testing.assert_allclose(x, 0.5 * np.ones(64), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
