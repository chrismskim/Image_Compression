"""Verify the Iteration 3 power-iteration SVD on simulated data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from methods import my_reconstruct, my_svd_power
from simulate import simulate_low_rank


def report(name, mine, reference):
    """Print maximum absolute and relative differences."""
    mine = np.asarray(mine, dtype=float)
    reference = np.asarray(reference, dtype=float)
    delta = mine - reference
    max_diff = np.max(np.abs(delta))
    reference_norm = np.sqrt(np.sum(reference**2))
    relative = np.sqrt(np.sum(delta**2)) / reference_norm if reference_norm else None
    relative_text = f"{relative:.3e}" if relative is not None else "n/a"
    print(f"{name:<30} max |diff| = {max_diff:.3e}   relative = {relative_text}")


def check_rank_five(A, U_ref, sigma_ref, Vt_ref):
    sigma, U, V = my_svd_power(A, k=5)
    signs = np.where(np.sum(U * U_ref[:, :5], axis=0) < 0, -1, 1)

    A5 = my_reconstruct(U, sigma, V, k=5)
    squared_error = np.sum((A - A5) ** 2)
    tail_energy = np.sum(sigma_ref[5:] ** 2)

    comparisons = [
        ("Top five singular values", sigma, sigma_ref[:5]),
        ("Left vectors, signs aligned", U * signs, U_ref[:, :5]),
        ("Right vectors, signs aligned", V * signs, Vt_ref[:5].T),
        ("Reconstruction vs. tail", squared_error, tail_energy),
    ]
    for name, mine, reference in comparisons:
        report(name, mine, reference)
        np.testing.assert_allclose(mine, reference, rtol=0, atol=1e-8)

    print(f"Rank-5 squared error: {squared_error:.12f}")
    print(f"Discarded energy:     {tail_energy:.12f}")
    print("All four rank-5 simulation checks passed.")


def check_stress_test(A, sigma_ref):
    print("\nRank-8 stress test")
    for i in range(4, 8):
        ratio = sigma_ref[i + 1] / sigma_ref[i]
        print(f"sigma_{i + 2}/sigma_{i + 1} = {ratio:.6f}; squared = {ratio**2:.6f}")

    try:
        sigma, U, V = my_svd_power(A, k=8)
    except RuntimeError as error:
        print(f"Rank-8 power iteration: {error}")
        return

    A8 = my_reconstruct(U, sigma, V, k=8)
    report("Top eight singular values", sigma, sigma_ref[:8])
    report("Rank-8 error vs. tail", np.sum((A - A8) ** 2), np.sum(sigma_ref[8:] ** 2))


def check_sample_sizes():
    row_counts = (120, 480, 1920)
    errors = np.empty((8, len(row_counts)))

    for seed in range(errors.shape[0]):
        noisy, clean = simulate_low_rank(m=row_counts[-1], seed=seed)
        for column, rows in enumerate(row_counts):
            _, _, V = my_svd_power(noisy[:rows], k=5)
            _, _, V_true = my_svd_power(clean[:rows], k=5)
            difference = V @ V.T - V_true @ V_true.T
            errors[seed, column] = np.sqrt(np.sum(difference**2))

    median_errors = np.median(errors, axis=0)
    print("\nMedian right-subspace error across eight seeds")
    for rows, error in zip(row_counts, median_errors):
        print(f"{rows:4d} rows: {error:.6e}")

    figures = Path(__file__).resolve().parents[1] / "figures"
    figures.mkdir(exist_ok=True)
    output = figures / "iteration3_simulation_sample_size.png"

    fig, ax = plt.subplots()
    ax.loglog(row_counts, median_errors, "o-")
    ax.set(
        xlabel="Number of rows (80 columns fixed)",
        ylabel="Median Frobenius distance between subspace projectors",
        title="Rank-5 signal subspace recovery",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"Saved plot: {output.name}")


def main():
    A, _ = simulate_low_rank()
    U_ref, sigma_ref, Vt_ref = np.linalg.svd(A, full_matrices=False)
    check_rank_five(A, U_ref, sigma_ref, Vt_ref)
    check_stress_test(A, sigma_ref)
    check_sample_sizes()


if __name__ == "__main__":
    main()