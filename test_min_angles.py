"""验证小波伴随、解析常量解，以及 PnP 的数据/去噪分离。"""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from run_min_angles import Wavelet,wavelet_tv,pnp_admm,bm3d_denoise
class MinimumAngleTests(unittest.TestCase):
    def test_wavelet_orthogonal_adjoint(self):
        rng=np.random.default_rng(7);W=Wavelet((64,64));u=rng.normal(size=(64,64));c=rng.normal(size=(64,64))
        np.testing.assert_allclose(W.transpose(W.forward(u)),u,atol=1e-12)
        self.assertAlmostEqual(np.sum(W.forward(u)*c),np.sum(u*W.transpose(c)),places=10)
        self.assertAlmostEqual(np.linalg.norm(W.forward(u)),np.linalg.norm(u),places=10)
    def test_wavelet_detail_constant_zero(self):
        W=Wavelet((64,64));np.testing.assert_allclose(W.forward(np.ones((64,64)))*W.mask,0,atol=1e-12)
    def test_wavelet_nonnegative_identity_constant(self):
        for value in (-.2,.7):
            x,_,_=wavelet_tv(sparse.eye(1024,format='csr'),np.full(1024,value),(32,32),1.,1.,.01,1.,250)
            np.testing.assert_allclose(x,max(value,0),atol=1e-10)
    def test_pnp_identity_denoiser_matches_data(self):
        # 去噪器设为恒等且非负时，非负恒定数据的固定点就是数据本身。
        b=np.full(1024,.3)
        with patch('run_min_angles.bm3d_denoise',side_effect=lambda x,s,p:np.maximum(x,0)):
            x,trace,_=pnp_admm(sparse.eye(1024,format='csr'),b,(32,32),1.,1.,.03,.01,20,np.zeros_like(b))
        np.testing.assert_allclose(x,b,atol=1e-6);self.assertEqual(trace[-1]['cg_not_converged_count'],0)
    def test_real_bm3d_finite_nonnegative(self):
        image=np.random.default_rng(3).normal(.3,.02,(32,32));x=bm3d_denoise(image,.02)
        self.assertEqual(x.shape,image.shape);self.assertTrue(np.isfinite(x).all());self.assertGreaterEqual(x.min(),0)
if __name__=='__main__':unittest.main()
