# DS5020 Final Project — Iteration 2

**Project Title:** Image Compression and Reconstruction Using Singular Value Decomposition (SVD)  
**Course:** DS5020 – Introduction to Linear Algebra  
**Semester:** Fall 2026

## Team Members

- **Minseong Kim** — Project Leader
- **Vaenkatakrishnan Srinivasan** — Team Member

## Project Overview

This project explores image compression and reconstruction using Singular Value Decomposition (SVD). We implement an SVD-based algorithm using Python and NumPy to decompose grayscale images and construct low-rank approximations.

We evaluate reconstruction errors, compression ratios, storage requirements, and energy retention at different rank values.

Our goal is to understand how the number of retained singular values affects image quality and storage efficiency.

## Roles and Responsibilities

| Task | Owner | Checker |
|---|---|---|
| Q1–Q3: SVD mathematical proofs | Vaenkatakrishnan Srinivasan | Minseong Kim |
| Q4–Q5: Frobenius norm, reconstruction error, storage cost | Minseong Kim | Vaenkatakrishnan Srinivasan |
| SVD implementation and mathematical algorithm | Vaenkatakrishnan Srinivasan | Minseong Kim |
| Real-image analysis and visualization | Minseong Kim | Vaenkatakrishnan Srinivasan |
| Simulation, definitions, and report integration | Both | Both |

## Project Structure

```text
Image-Compression-SVD/
├── data/
│   └── grace_hopper.jpg
├── docs/
│   ├── Iteration2_Final_Report.pdf
│   ├── DS5020_Project_Plan.xlsx
│   ├── real_image_results.csv
│   └── simulation_results.csv
├── figures/
│   ├── real_energy.png
│   ├── real_reconstructions.png
│   ├── simulation_error.png
│   └── simulation_spectrum.png
├── src/
│   ├── load_image.py
│   ├── simulate.py
│   └── svd_core.py
├── tests/
│   └── test_svd.py
├── README.md
└── requirements.txt
```

## Installation

Install Python 3.10 or later and the required libraries.

```bash
python3 -m pip install -r requirements.txt
```

## Running the Project

### 1. Rank-5 Simulation

Generate a synthetic matrix with a rank-5 signal and Gaussian noise.

```bash
python3 src/simulate.py
```

The simulation verifies the theoretical properties of SVD, including reconstruction errors, singular values, and orthonormality.

### 2. Real Image Analysis

Apply SVD-based compression to a grayscale image.

```bash
python3 src/load_image.py
```

The script generates reconstructed images at different rank values and calculates reconstruction errors, compression ratios, and energy retention.

### 3. Run Tests

```bash
python3 -m unittest discover -s tests -v
```

## Dataset

We use the Grace Hopper sample photograph provided by Matplotlib.

The original RGB image is converted to grayscale and represented as a matrix of size 600 × 512.

Each matrix entry represents the intensity of a pixel.

## Simulation

We generate a 120 × 80 matrix with a rank-5 signal and Gaussian noise using a fixed random seed of 0.

We evaluate the reconstruction at ranks 1, 3, 5, and 10.

| Rank | Squared Reconstruction Error | Stored Values |
|---|---:|---:|
| 1 | 30,397.1176 | 201 |
| 3 | 11,318.1041 | 603 |
| 5 | 0.8730 | 1,005 |
| 10 | 0.7077 | 2,010 |

The results demonstrate that increasing the approximation rank reduces reconstruction error.

## Real Image Results

We evaluate the relationship between rank, energy retention, and compression ratio.

| Rank | Energy Retention | Compression Ratio |
|---|---:|---:|
| 5 | 88.00% | 55.20× |
| 10 | 92.97% | 27.60× |
| 20 | 96.32% | 13.80× |
| 40 | 98.38% | 6.90× |

The minimum rank required to retain at least 95% of the image's Frobenius energy is **15**.

Compression ratios represent theoretical storage savings based on the number of stored scalar values, rather than actual encoded image file sizes.

## Visualizations

The `figures/` folder contains:

- `real_reconstructions.png` — Original and reconstructed images at different ranks.
- `real_energy.png` — Energy retention versus rank.
- `simulation_spectrum.png` — Singular value spectrum of the simulated matrix.
- `simulation_error.png` — Reconstruction error comparison.

## Implementation Notes

Our implementation constructs SVD from the eigendecomposition of the Gram matrix using `numpy.linalg.eigh`.

We do not use NumPy's built-in SVD routine to construct the decomposition. The built-in routine is used only for numerical comparison.

## Final Report

The complete mathematical derivations, assumptions, algorithms, and experimental results are documented in our Overleaf report.

[View Iteration 2 Final Report](docs/DS5020_Iteration2_Report.pdf)

## Submission

The final project deliverables consist of:

- Overleaf report containing mathematical definitions, assumptions, Q1–Q5 derivations, algorithms, and simulation results.
- GitHub repository containing Python code, data, figures, and documentation.
- Updated Excel Project Plan.

Each team member submits the GitHub repository link and Overleaf read-only link individually through Canvas.
