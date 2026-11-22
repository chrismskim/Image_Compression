"""Reproducible rank-five simulation with Gaussian noise."""
from pathlib import Path
import numpy as np

def simulate_low_rank(m=120, n=80, r=5, noise=0.01, seed=0):
    """Return the noisy matrix A and the clean rank-r signal A_true."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(m, r))
    C = rng.normal(size=(r, n))
    A_true = B @ C
    A = A_true + noise * rng.normal(size=(m, n))
    return A, A_true


def main():
    """Run the Iteration 2 analysis and generate its report outputs."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if __package__:
        from .svd_core import metrics, reconstruct, svd_from_scratch
    else:
        from svd_core import metrics, reconstruct, svd_from_scratch

    base = Path(__file__).resolve().parents[1]
    figures = base / "figures"
    docs = base / "docs"
    figures.mkdir(exist_ok=True)

    A, A_true = simulate_low_rank()
    U, s, Vt = svd_from_scratch(A)
    s_builtin = np.linalg.svd(A, compute_uv=False)  # Comparison only.

    q4_error = abs(np.sum(A**2) - np.sum(s**2))
    U5 = U[:, :5]
    orth_error = np.max(np.abs(U5.T @ U5 - np.eye(5)))

    rows = []
    for k in (1, 3, 5, 10):
        A_k = reconstruct(U, s, Vt, k)
        rows.append(metrics(A, A_k, s, k))

    results = pd.DataFrame(rows)
    results.to_csv(docs / "simulation_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(s) + 1), s, "o-", markersize=3)
    ax.axvline(5, linestyle="--", color="gray", label="True signal rank = 5")
    ax.set(
        xlabel="Component index",
        ylabel="Singular value",
        title="Simulated Rank-5 Matrix: Singular Values",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "simulation_spectrum.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        results["rank"], results["sq_error"], "o-",
        label="Actual squared Frobenius error",
    )
    ax.plot(
        results["rank"], results["tail_energy"], "x--",
        label="Discarded singular-value energy",
    )
    ax.set(
        xlabel="Retained rank k",
        ylabel="Squared error",
        title="Q5: Reconstruction Error Identity",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "simulation_error.png", dpi=170)
    plt.close(fig)

    U_true, s_true, Vt_true = svd_from_scratch(A_true)
    A_true_5 = reconstruct(U_true, s_true, Vt_true, 5)
    rank5_error = np.linalg.norm(A_true - A_true_5, "fro")

    with (docs / "simulation_generated.tex").open("w", encoding="utf-8") as f:
        f.write(r"\newcommand{\SimTraceError}{" + f"{q4_error:.2e}" + "}\n")
        f.write(r"\newcommand{\SimOrthError}{" + f"{orth_error:.2e}" + "}\n")
        f.write(r"\newcommand{\SimNoiselessError}{" + f"{rank5_error:.2e}" + "}\n")
        f.write(r"\newcommand{\SimTable}{" + "\n")
        f.write(r"\begin{tabular}{rrrrr}\toprule" + "\n")
        f.write(
            r"Rank $k$ & Measured $\|A-A_k\|_F^2$ & "
            r"$\sum_{i>k}\sigma_i^2$ & Scalars & Energy (\%) \\"
            + "\n"
        )
        f.write(r"\midrule" + "\n")
        for row in rows:
            f.write(
                f"{row['rank']} & {row['sq_error']:.4f} & "
                f"{row['tail_energy']:.4f} & {row['stored_scalars']} & "
                f"{row['energy_percent']:.2f} " + r"\\" + "\n"
            )
        f.write(r"\bottomrule\end{tabular}" + "\n}\n")

    print(
        "Simulation: eigenvalue/SVD Q4 difference", q4_error,
        "Q2 orthonormality max error", orth_error,
    )
    print(
        "Noiseless rank-5 error", rank5_error,
        "Built-in singular values max difference", np.max(np.abs(s - s_builtin)),
    )
    print(results.to_string(index=False))

    assert np.allclose(results["sq_error"], results["tail_energy"], rtol=1e-8, atol=1e-7)
    assert q4_error < 1e-6 and orth_error < 1e-6


if __name__ == "__main__":
    main()
