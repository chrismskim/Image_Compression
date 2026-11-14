"""Smoke tests for decomposition and SVD compression identities."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from svd_core import svd_from_scratch, reconstruct, metrics

class TestSVD(unittest.TestCase):
    def test_tall_and_wide(self):
        rng = np.random.default_rng(17)
        for shape in [(20, 9), (9, 20)]:
            A = rng.normal(size=shape)
            U, s, Vt = svd_from_scratch(A)
            self.assertTrue(np.allclose(reconstruct(U,s,Vt,len(s)), A, atol=1e-8))
            self.assertTrue(np.allclose(U.T @ U, np.eye(len(s)), atol=1e-8))
            self.assertTrue(np.allclose(Vt @ Vt.T, np.eye(len(s)), atol=1e-8))
            for k in [1, 3, min(shape)]:
                Ak = reconstruct(U,s,Vt,k)
                out = metrics(A, Ak, s, k)
                self.assertAlmostEqual(out['sq_error'],out['tail_energy'],places=7)
    def test_storage(self):
        A = np.diag([5.0, 2.0, 1.0])
        U,s,Vt = svd_from_scratch(A)
        d = metrics(A, reconstruct(U,s,Vt,2), s, 2)
        self.assertEqual(d['stored_scalars'], 2*(3+3+1))
        self.assertAlmostEqual(d['sq_error'], 1.0)
if __name__ == '__main__': unittest.main()
