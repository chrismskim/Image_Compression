"""Verify the Iteration 3 SVD on the Grace Hopper photograph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from check_simulation import report
from methods import my_reconstruct, my_svd_power


def load_grayscale(path):
    pixels = plt.imread(path)
    gray = pixels[:, :, :3] @ np.array([0.299, 0.587, 0.114])
    return gray / 255.0


def main():
    root = Path(__file__).resolve().parents[1]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    A = load_grayscale(root / "data" / "grace_hopper.jpg")
    m, n = A.shape
    print(f"Image shape: {m} x {n}")
    print(f"Nonfinite pixels: {np.count_nonzero(~np.isfinite(A))}")
    print(f"Pixel range: [{A.min():.6f}, {A.max():.6f}]")

    total_energy = np.sum(A**2)
    sigma, U, V = my_svd_power(A, k=50)
    energy = np.cumsum(sigma**2) / total_energy

    if energy[-1] < 0.95:
        raise RuntimeError("The first 50 components retain less than 95% of the energy.")
    k95 = int(np.searchsorted(energy, 0.95)) + 1
    print(f"Smallest rank retaining 95% energy: {k95}")
    if k95 > 1:
        print(f"Energy at rank {k95 - 1}: {energy[k95 - 2]:.4%}")
    print(f"Energy at rank {k95}: {energy[k95 - 1]:.4%}")

    U_ref, sigma_ref, Vt_ref = np.linalg.svd(A, full_matrices=False)
    signs = np.where(np.sum(U * U_ref[:, :50], axis=0) < 0, -1, 1)
    report("Top 50 singular values", sigma, sigma_ref[:50])
    report("Top five U, signs aligned", U[:, :5] * signs[:5], U_ref[:, :5])
    report("Top five V, signs aligned", V[:, :5] * signs[:5], Vt_ref[:5].T)
    report("U orthogonality", U.T @ U, np.eye(50))
    report("V orthogonality", V.T @ V, np.eye(50))

    ranks = (5, 20, 50, k95)
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(15, 4))
    axes[0].imshow(A, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    print(f"\nOriginal storage: {m * n:,} scalars")

    for ax, k in zip(axes[1:], ranks):
        A_k = my_reconstruct(U, sigma, V, k)
        A_ref_k = (U_ref[:, :k] * sigma_ref[:k]) @ Vt_ref[:k]
        stored = k * (m + n + 1)
        squared_error = np.sum((A - A_k) ** 2)
        print(
            f"k={k:2d} | energy={energy[k - 1]:.4%} | "
            f"storage={stored:,} scalars | squared error={squared_error:.6f}"
        )
        report(f"Rank-{k} vs. NumPy", A_k, A_ref_k)
        ax.imshow(np.clip(A_k, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k = {k}")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "iteration3_real_reconstructions.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(energy) + 1), 100 * energy)
    ax.axhline(95, color="gray", linestyle="--")
    ax.axvline(k95, color="gray", linestyle=":")
    ax.set(
        xlabel="Rank k",
        ylabel="Retained energy (%)",
        title=f"Grace Hopper photograph: 95% energy at k = {k95}",
    )
    fig.tight_layout()
    fig.savefig(figures / "iteration3_real_energy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()