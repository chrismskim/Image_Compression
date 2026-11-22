"""Work through the Iteration 3 two-by-two SVD example."""

from fractions import Fraction

import numpy as np

from methods import my_deflate, my_gram_matrix, my_reconstruct, my_svd_power


def main():
    A = np.array([[3, 0], [4, 5]], dtype=float)
    G = my_gram_matrix(A)
    trace = int(G[0, 0] + G[1, 1])
    determinant = int(G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])

    print("A:")
    print(A)
    print("\nG = A.T @ A:")
    print(G)
    print(f"\nCharacteristic polynomial: lambda^2 - {trace} lambda + {determinant}")
    print("Factored: (lambda - 45)(lambda - 5)")

    sigma, U, V = my_svd_power(A, k=2)
    eigenvalues = sigma**2
    print("\nEigenvalues:", eigenvalues)
    print("Right singular vectors V (columns):")
    print(V)
    print("Gram matrix after removing the first eigenpair:")
    print(my_deflate(G, eigenvalues[0], V[:, 0]))
    print("Singular values:", sigma)
    print("A @ V:")
    print(A @ V)
    print("Left singular vectors U = (A @ V) / sigma:")
    print(U)

    A1 = my_reconstruct(U, sigma, V, k=1)
    residual = A - A1
    squared_error = np.sum(residual**2)
    print("\nRank-1 approximation A1:")
    print(A1)
    print("Residual A - A1:")
    print(residual)
    print(f"Squared Frobenius error: {squared_error:.12g}")

    # Exactly, v1 v1^T is the 2-by-2 matrix with every entry equal to 1/2.
    # Thus A1 = A v1 v1^T has half of each row sum in both columns.
    exact_A1 = [
        [Fraction(int(row.sum()), 2)] * 2
        for row in A
    ]
    exact_error = sum(
        (Fraction(int(A[i, j])) - exact_A1[i][j]) ** 2
        for i in range(2)
        for j in range(2)
    )
    print("Exact A1 rows:", [[str(value) for value in row] for row in exact_A1])
    print("Exact squared error:", exact_error)

    tolerance = {"rtol": 0, "atol": 1e-10}
    np.testing.assert_allclose(eigenvalues, [45, 5], **tolerance)
    np.testing.assert_allclose(G @ V, V * eigenvalues, **tolerance)
    np.testing.assert_allclose(A.T @ U, V * sigma, **tolerance)
    np.testing.assert_allclose(A1, np.array(exact_A1, dtype=float), **tolerance)
    np.testing.assert_allclose(squared_error, sigma[1] ** 2, **tolerance)
    np.testing.assert_allclose(squared_error, float(exact_error), **tolerance)
    np.testing.assert_allclose(my_reconstruct(U, sigma, V, k=2), A, **tolerance)
    print("\nAll worked-example checks passed.")


if __name__ == "__main__":
    main()