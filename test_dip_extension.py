# 网络数值测试：检查展开顺序、稀疏投影梯度、TV 和配对初始化。
# CUDA 不可用时 GPU 专用测试会跳过；单元测试通过不等于完整训练已经完成。
#
"""Numerical contracts used by the neural CT extension."""
import unittest
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch
import numpy as np
from scipy import sparse
from run_dip_extension import image_vector, sparse_tensor, SparseProjection, mean_tv, initialization, observation_scale


# 检查网络依赖的数值约定，避免把错误梯度或错误像素顺序带入长时间训练。
class DIPContracts(unittest.TestCase):
    # 对照解析梯度 A.T@(A@x-b)/||b||²，检查 GPU 投影反向传播。
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_projection_gradient(self):
        rng = np.random.default_rng(42)
        dense = rng.normal(size=(17, 31)).astype(np.float32)
        dense[np.abs(dense) < 1] = 0
        A = sparse.csr_matrix(dense)
        x = rng.normal(size=(31, 1)).astype(np.float32)
        b = rng.normal(size=(17, 1)).astype(np.float32)
        leaf = torch.tensor(x, device="cuda", requires_grad=True)
        projection = SparseProjection.apply(leaf, sparse_tensor(A, "cuda"), sparse_tensor(A.T, "cuda"))
        np.testing.assert_allclose(projection.detach().cpu().numpy(), A @ x, rtol=1e-5, atol=1e-5)
        energy = float(np.sum(b**2))
        loss = (projection - torch.tensor(b, device="cuda")).square().sum() / (2 * energy)
        loss.backward()
        expected = A.T @ (A @ x - b) / energy
        relative_error = np.linalg.norm(leaf.grad.cpu().numpy() - expected) / np.linalg.norm(expected)
        self.assertLess(relative_error, 1e-4)

    # 用小图像检查 F 顺序展开、前向 A 和反向 A.T 的一致性。
    def test_column_major_projection_and_adjoint(self):
        rng = np.random.default_rng(27)
        values = rng.normal(size=(6, 4))
        values[np.abs(values) < 0.8] = 0
        values[1] = 0  # Noise-only rows remain in the data loss.
        A = sparse.csr_matrix(values)
        truth_image = np.array([[1., 2.], [3., 4.]])
        image = torch.tensor(truth_image.reshape(1, 1, 2, 2), requires_grad=True)
        vector = image_vector(image)
        np.testing.assert_array_equal(vector.detach().numpy().ravel(), truth_image.reshape(-1, order="F"))
        tA, tAT = sparse_tensor(A, "cpu", torch.float64), sparse_tensor(A.T, "cpu", torch.float64)
        y = SparseProjection.apply(vector, tA, tAT)
        np.testing.assert_allclose(y.detach().numpy().ravel(), A @ truth_image.reshape(-1, order="F"), atol=1e-12)
        (y.square().sum() / 2).backward()
        expected = (A.T @ (A @ truth_image.reshape(-1, order="F"))).reshape(2, 2, order="F")
        np.testing.assert_allclose(image.grad.numpy()[0, 0], expected, atol=1e-12)

    # 常数图像平滑 TV 为零，梯度也必须有限且为零。
    def test_tv_constant_has_finite_zero_gradient(self):
        image = torch.ones(1, 1, 8, 8, requires_grad=True)
        tv = mean_tv(image)
        tv.backward()
        self.assertEqual(float(tv.detach()), 0)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertEqual(float(image.grad.abs().max()), 0)

    # 相同种子必须生成相同权重与输入；幅度尺度只使用 A、b。
    def test_paired_initialization_and_observation_scale(self):
        net1, noise1, digest1 = initialization(7, 64, "cpu")
        net2, noise2, digest2 = initialization(7, 64, "cpu")
        self.assertEqual(digest1, digest2)
        self.assertTrue(torch.equal(noise1, noise2))
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        A = sparse.eye(4).tocsr()
        self.assertAlmostEqual(observation_scale(A, np.ones(4) * 0.03), 0.03)


if __name__ == "__main__":
    unittest.main()
