"""SVD from symmetric eigendecomposition. No np.linalg.svd used in the algorithm."""
import numpy as np

def svd_from_scratch(A, tol=None):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    # Work with the smaller Gram matrix for numerical efficiency.
    if m >= n:
        eigenvalues, V = np.linalg.eigh(A.T @ A)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, V = eigenvalues[order], V[:, order]
        singular_values = np.sqrt(np.clip(eigenvalues, 0, None))
        tolerance = (tol if tol is not None else max(m, n)*np.finfo(float).eps*singular_values[0]) if singular_values.size else 0
        r = int(np.sum(singular_values > tolerance))
        s, V = singular_values[:r], V[:, :r]
        U = (A @ V) / s
    else:
        eigenvalues, U = np.linalg.eigh(A @ A.T)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, U = eigenvalues[order], U[:, order]
        singular_values = np.sqrt(np.clip(eigenvalues, 0, None))
        tolerance = (tol if tol is not None else max(m, n)*np.finfo(float).eps*singular_values[0]) if singular_values.size else 0
        r = int(np.sum(singular_values > tolerance))
        s, U = singular_values[:r], U[:, :r]
        V = (A.T @ U) / s
    return U, s, V.T

def reconstruct(U, s, Vt, k):
    if not 0 <= k <= len(s):
        raise ValueError('k must be in [0, rank]')
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def metrics(A, Ak, s, k):
    m, n = A.shape
    sqerr = float(np.sum((A - Ak)**2))
    tail = float(np.sum(s[k:]**2))
    original = m*n
    stored = k*(m+n+1)
    return {
        'rank': k, 'sq_error': sqerr, 'tail_energy': tail,
        'fro_error': np.sqrt(sqerr), 'stored_scalars': stored,
        'compression_ratio': original/stored if stored else float('inf'),
        'energy_percent': 100*float(np.sum(s[:k]**2))/float(np.sum(s**2)) if np.sum(s**2)>0 else 0,
    }
