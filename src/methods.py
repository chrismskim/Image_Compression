import numpy as np


# G = A^T A
def my_gram_matrix(A):
    """Compute the Gram matrix of a real matrix A."""
    A = np.asarray(A, dtype=float)
    return A.T @ A


# v_next = Gv / ||Gv||_2, lambda = v^T Gv
def my_power_iteration(G, start, tol=1e-13, max_iter=20_000):
    """Return an eigenvalue, its unit eigenvector, and the iteration count."""
    v = np.asarray(start, dtype=float)
    length = np.sqrt(v @ v)
    if length == 0:
        raise ValueError("The starting vector must be nonzero.")
    v = v / length

    scale = np.sqrt(np.sum(G * G))
    for step in range(1, max_iter + 1):
        w = G @ v
        length = np.sqrt(w @ w)
        if length == 0:
            raise ValueError("The matrix maps the starting vector to zero.")

        v = w / length
        Gv = G @ v
        eigenvalue = float(v @ Gv)
        residual = Gv - eigenvalue * v

        if np.sqrt(residual @ residual) <= tol * scale:
            return eigenvalue, v, step

    raise RuntimeError(f"Power iteration did not converge in {max_iter} steps.")

# G_next = G - lambda * v * v^T
def my_deflate(G, eigenvalue, v):
    """Subtract the eigenpair's contribution from the Gram matrix."""
    return G - eigenvalue * np.outer(v, v)

# sigma_i = sqrt(lambda_i)
def my_singular_values(eigenvalues):
    """Convert positive Gram eigenvalues to singular values."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if np.any(eigenvalues <= 0):
        raise ValueError("This step requires positive Gram eigenvalues.")
    return np.sqrt(eigenvalues)


# U = (A V) / sigma, dividing each column by its singular value
def my_left_vectors(A, V, sigma):
    """Compute the left singular vectors from the right singular vectors."""
    return (np.asarray(A, dtype=float) @ V) / sigma


# A_k = U_k diag(sigma_1, ..., sigma_k) V_k^T
def my_reconstruct(U, sigma, V, k):
    """Reconstruct the matrix using its first k singular components."""
    return (U[:, :k] * sigma[:k]) @ V[:, :k].T

# A_k = U_k diag(sigma) V_k^T, where A^T A v_i = sigma_i^2 v_i
def my_svd_power(A, k, seed=0):
    """Return the leading k singular values and the columns of U and V."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("A must be a finite two-dimensional matrix.")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= min(A.shape):
        raise ValueError("k must be an integer from 1 to min(A.shape).")

    G = my_gram_matrix(A)
    gram_scale = np.sqrt(np.sum(G * G))
    rank_tol = np.finfo(float).eps * max(A.shape) * gram_scale
    rng = np.random.default_rng(seed)

    eigenvalues = []
    right_vectors = []
    for _ in range(k):
        if np.sqrt(np.sum(G * G)) <= rank_tol:
            raise ValueError("k exceeds the numerical rank of A.")

        start = rng.standard_normal(A.shape[1])
        eigenvalue, v, _ = my_power_iteration(G, start)
        if eigenvalue <= rank_tol:
            raise ValueError("k exceeds the numerical rank of A.")

        eigenvalues.append(eigenvalue)
        right_vectors.append(v)
        G = my_deflate(G, eigenvalue, v)

    sigma = my_singular_values(eigenvalues)
    V = np.column_stack(right_vectors)
    U = my_left_vectors(A, V, sigma)
    return sigma, U, V