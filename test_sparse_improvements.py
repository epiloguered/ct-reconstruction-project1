"""数值测试：TGV 伴随、原始-对偶解和投影验证集隔离。"""
import unittest
import numpy as np
from scipy import sparse
from run_sparse_improvements import gradient,adjoint,symmetric_gradient,symmetric_adjoint,primal_dual,split_observed
class ImprovementTests(unittest.TestCase):
    def test_gradient_adjoint(self):
        rng=np.random.default_rng(4);u=rng.normal(size=(9,7));p=rng.normal(size=(2,9,7))
        self.assertAlmostEqual(float(np.sum(gradient(u)*p)),float(np.sum(u*adjoint(p))),places=11)
    def test_symmetric_gradient_adjoint(self):
        rng=np.random.default_rng(5);v=rng.normal(size=(2,9,7));q=rng.normal(size=(3,9,7))
        self.assertAlmostEqual(float(np.sum(symmetric_gradient(v)*q)),float(np.sum(v*symmetric_adjoint(q))),places=11)
    def test_nonnegative_constant_identity(self):
        A=sparse.eye(64,format="csr")
        for value in (-0.2,0.7):
            for ratio in (0,2):
                x,_,_=primal_dual(A,np.full(64,value),(8,8),1.,1.,0.01,ratio,True,iterations=250)
                np.testing.assert_allclose(x,np.full(64,max(value,0)),atol=1e-10)
    def test_unconstrained_constant_identity(self):
        x,_,_=primal_dual(sparse.eye(64,format="csr"),np.full(64,-0.2),(8,8),1.,1.,0.01,0,False,iterations=250)
        np.testing.assert_allclose(x,-0.2,atol=1e-10)
    def test_heldout_complete_disjoint_blocks(self):
        for angles in (15,30):
            A=sparse.csr_matrix((angles*7,64));b=np.arange(angles*7)
            train,val=split_observed(A,b,angles)
            self.assertEqual(np.intersect1d(train,val).size,0)
            np.testing.assert_array_equal(np.sort(np.r_[train,val]),np.arange(A.shape[0]))
            self.assertEqual(len(val),A.shape[0]//5)
            for row in val:self.assertTrue(all(r in val for r in range(row//7*7,row//7*7+7)))
if __name__=="__main__":unittest.main()
