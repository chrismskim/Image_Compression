# DS5020 Final Project — Image Compression with SVD

**Project title:** Image Compression and Reconstruction Using Singular Value Decomposition (SVD)  
**Course:** DS5020 – Introduction to Linear Algebra  
**Semester:** Fall 2026

## Team Members

- **Minseong Kim** — Iteration 2 project leader
- **Vaenkatakrishnan Srinivasan** — Iteration 3 Project Leader

## Project Overview

We use the singular value decomposition to reconstruct a grayscale image at different ranks. We measure how the number of retained singular components affects reconstruction error, energy retention, and the number of scalars needed to store the approximation.

In **Iteration 2**, we derived the mathematical identities and evaluated them on a synthetic matrix and a real image. In **Iteration 3**, we implemented the SVD using power iteration with deflation and verified it on an exact worked example, a seeded simulation, and the Grace Hopper photograph.

## Installation

Run these commands from the repository root in Windows PowerShell. If you already have a `.venv` with the dependencies installed, skip to the run commands.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running Iteration 3

```powershell
.\.venv\Scripts\python.exe .\src\small_example.py
.\.venv\Scripts\python.exe .\src\check_simulation.py
.\.venv\Scripts\python.exe .\src\run_real.py
```

| Script | What it does |
|---|---|
| `src/small_example.py` | Prints the SVD steps for the assigned 2 × 2 matrix and verifies the rank-1 approximation using exact fractions and substitution. |
| `src/check_simulation.py` | Checks rank 5 against NumPy, runs the rank-8 convergence stress test, and measures signal subspace recovery at three sample sizes. |
| `src/run_real.py` | Reconstructs the Grace Hopper image at the required ranks, checks the results against NumPy, and plots retained energy. |

Run the repository’s existing unit tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Iteration 3 Results

### Worked Example

For the assigned matrix `[[3, 0], [4, 5]]`, the Gram eigenvalues are **45** and **5**, giving singular values \(3\sqrt{5}\) and \(\sqrt{5}\). The rank-1 approximation has rows `[1.5, 1.5]` and `[4.5, 4.5]`. Its squared Frobenius error is exactly **5**, equal to the square of the discarded singular value.

### Rank-5 Simulation

The baseline is a **120 × 80** matrix with a rank-5 signal, Gaussian noise with standard deviation **0.01**, and random seed **0**. The rank-5 squared reconstruction error and discarded spectral energy both round to **0.873022217597**.

| Comparison with NumPy | Maximum absolute difference |
|---|---:|
| Top five singular values | 8.527 × 10⁻¹⁴ |
| Left singular vectors after sign alignment | 5.110 × 10⁻¹³ |
| Right singular vectors after sign alignment | 4.780 × 10⁻¹³ |
| Squared reconstruction error vs. discarded energy | 1.110 × 10⁻¹⁵ |

The rank-8 stress test probes the closely spaced singular values in the noise spectrum. The sample-size experiment holds the number of columns fixed at 80. Across eight seeds, the median right-subspace error was **2.994 × 10⁻³** at 120 rows, **1.458 × 10⁻³** at 480 rows, and **7.371 × 10⁻⁴** at 1,920 rows.

### Real Image

We use the Grace Hopper sample photograph from Matplotlib. We convert the RGB image to grayscale and scale it to \([0,1]\), giving a **600 × 512** matrix with **307,200** pixel values.

| Rank | Energy retained | Stored scalars |
|---:|---:|---:|
| 5 | 87.9961% | 5,565 |
| 15 | 95.1240% | 16,695 |
| 20 | 96.3187% | 22,260 |
| 50 | 98.8120% | 55,650 |

The smallest rank retaining at least 95% of the energy is **15**. Rank 14 retains **94.7804%**. At rank 15, the original-to-factor storage ratio is **18.40×**, measured by scalar counts. The top 50 singular values agreed with NumPy to a maximum absolute difference of **7.105 × 10⁻¹⁴**.

## Iteration 3 Figures

- [Rank-5 recovery versus sample size](figures/iteration3_simulation_sample_size.png)
- [Grace Hopper rank-k reconstructions](figures/iteration3_real_reconstructions.png)
- [Grace Hopper retained energy](figures/iteration3_real_energy.png)

## Implementation Notes

- `src/methods.py` implements the Iteration 3 SVD using power iteration with deflation on the Gram matrix. The verification scripts use `numpy.linalg.svd` as a reference.
- `src/simulate.py` provides `simulate_low_rank()` for the Iteration 3 checks and can still run the original Iteration 2 simulation.
- `src/svd_core.py` contains the Iteration 2 implementation, which uses `numpy.linalg.eigh`. `src/load_image.py` runs the original Iteration 2 image analysis.

## Reports and Project Plan

- [Iteration 2 report](docs/DS5020_Iteration2_Report.pdf)
- [Iteration 3 Report](docs/DS5020_Iteration3_Report.pdf)
- [Project plan](docs/DS5020_Project_Plan.xlsx)