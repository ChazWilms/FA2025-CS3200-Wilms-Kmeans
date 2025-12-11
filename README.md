# Optimizing K-Means Clustering with MPI

**Course:** CS 3200: Introduction to Machine Learning (Fall 2025)  
**Author:** Chaz Wilms  
**Tech Stack:** C++, OpenMPI, Python (pandas/scikit-learn)

---

## 📌 Project Overview
This project explores the parallelization of the **K-Means Clustering** algorithm to improve computational efficiency on large datasets. By utilizing the **Message Passing Interface (MPI)**, the workload is distributed across multiple processor cores (Master-Worker model), significantly reducing runtime compared to traditional serial implementations.

Key objectives achieved:
* **Parallel Implementation:** Custom C++ K-Means using `MPI_Bcast` and `MPI_Allreduce`.
* **Scalability Analysis:** Benchmarked on 100,000+ synthetic data points to measure Speedup and Efficiency.
* **Real-World Application:** Applied clustering to a large-scale Customer Segmentation dataset to validate model accuracy (SSE) on real attributes.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `serial_kmeans.cpp` | Baseline serial implementation of Lloyd's algorithm. |
| `mpi_kmeans.cpp` | Parallel implementation using OpenMPI (Master-Worker). |
| `data_setup.py` | Python script to generate synthetic data (100k) and clean real-world data. |
| `plot_results.py` | Generates Speedup and Convergence graphs for the report. |
| `synthetic_data.csv` | Generated dataset for performance benchmarking. |
| `large_real_data.csv` | Processed real-world dataset (Age, Income, Score). |

---

## 🚀 Getting Started

### Prerequisites
* **C++ Compiler:** `g++` or `clang++` (supporting C++17).
* **MPI Library:** OpenMPI (Install via `brew install open-mpi` on macOS).
* **Python 3:** Required for data generation and plotting.
    * Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

### 1. Data Preparation
Run the setup script to generate the synthetic benchmark data and clean the real-world CSV:
```bash
python3 data_setup.py
