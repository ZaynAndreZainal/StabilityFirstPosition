# StabilityFirstPoistion: Codebase for NeruIPS 2026 Postion Paper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation and experimental data for the paper:

**"Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design"**

*Zayn Andre Zainal, Omid Kavahei, Isabelle Aguilar, Luis Fernando Herbozo CorteZhaojing Huang, , *

## Abstract

> Despite rapid gains in biomedical predictive performance, current evaluation and training practice still defaults to Independent and Identically Distributed (IID), fixed-step assumptions that are poorly matched to continuous-time physiology, and safety-critical deployment. This position paper argues that safety-relevant biomedical Machine Learning (ML) should adopt a stability-first design and reporting standard: irregular-time stress testing plus hazard-linked training diagnostics. We highlight two structural hazards: (i) discretisation-by-default can entangle physiology with the observation process (the Discrete-Time Fallacy); and (ii) training can persist in Edge-of-Stability (EoS) regimes where $\rho(t)=\eta\lambda_{\max}(t)/2>1$, producing perturbation-sensitive solutions. We provide controlled mechanism probes and a minimal reporting standard to guide domain-specific validation. Our goal is not to mandate continuous-time models, but to make shift sensitivity and optimisation instability more verifiable before deployment, escalating only when diagnostics and robustness curves justify the cost.
---

## Repository Structure

The codebase is organized by experiment, corresponding directly to the Appendices in the paper:

```text
OptimizationTheoryGap/
├── experiments/
│   ├── 01_edge_of_stability/    # [Appendix C.1] Optimisation Instability (Edge of Stability) 
│   ├── 02_irregular_sampling/   # [Appendix C.2] Topological Stability and Irregular Sampling
│   └── 03_optimizer_compare/    # [Appendix C.3] End-to-End Optimiser Benchmark and Pareto Trade-off Plots
├── requirements.txt             # Shared dependencies for all experiments
└── README.md                    # You are here
```


---

## Experiments (mapped to the paper)

### 1. Edge of Stability Visualisation (Appendix C.1)

*Located in: **`experiments/01_edge_of_stability/`***

Investigates optimisation instability in deep networks trained on biomedical data (MedMNIST / PathMNIST).

- **Key finding**: Training can “surf” high-curvature regions where $\lambda_{\max} > 2/\eta$ (equivalently $\rho(t)=\eta\lambda_{\max}(t)/2>1$), implying perturbation-sensitive solutions despite decreasing loss.
- **Outputs**: Training loss vs sharpness trajectories, 3D loss-landscape slices, and optimisation-path projections.
- **Primary metrics**: $\lambda_{\max}$ (Hessian spectral proxy), $\rho(t)$, and summary statistics across seeds.

### 2. Irregular Sampling Benchmark (Appendix C.2)

*Located in: **`experiments/02_irregular_sampling/`***

Demonstrates the **Discrete-Time Fallacy** using FitzHugh–Nagumo dynamics, benchmarking Neural ODEs against time-aware discrete baselines (e.g., T-LSTM and ODE-RNN).

- **Key finding**: Discrete RNN baselines can accumulate phase error and drift under irregular resampling, while Neural ODEs can remain phase-locked by learning a stable continuous-time vector field.
- **Primary metrics**: RMSE, phase-space divergence, and long-horizon error boundedness under schedule perturbations (jitter / thinning / bursty missingness).

### 3. Optimiser Benchmark + Pareto Trade-off Plots (Appendix C.3)  **(NEW)**

*Located in: **`experiments/03_optimizer_compare/`***

Adds an end-to-end optimiser comparison benchmark (e.g., SGD, Momentum, Adam, AdamW, RMSprop, Lion, SAM variants) and visualises **efficiency–performance trade-offs**.

- **Key finding**: Curvature-aware methods can improve final IID performance and/or robustness proxies in some regimes but may increase time-to-target. Results should be interpreted on an efficiency–reliability frontier rather than IID score alone.
- **Outputs**: Pareto/frontier plots (performance vs time-to-target / wall-clock), plus per-optimiser summary tables.
- **Primary metrics**: IID performance (Accuracy/AUC), wall-clock runtime, time-to-target, and (optionally) failure rate across seeds.

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. We recommend creating a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (PyTorch, SciPy, Matplotlib, MedMNIST)
pip install -r requirements.txt
```

_Note: For GPU acceleration, please install the appropriate CUDA version of PyTorch from pytorch.org before running the requirements file._

### Experiment C.1: Edge of Stability (MedMNIST)

To reproduce the loss landscape visualization and Figure 3 from the paper:
```bash
cd experiments/01_edge_of_stability

# 1. Train the model (run for multiple seeds for confidence intervals)
python3 train_eos.py --seed 0

# 2. Visualize the trajectory
python3 visualize.py --checkpoint 0

# 3. Visualize multi-seed trajectory (publication figure)
python3 visualize_multi.py --seed 0
```

_Output: `figures/eos_single_0.png` showing the 3D optimization path._

### Experiment C.2: Irregular Sampling (FitzHugh-Nagumo)

To reproduce the time-series benchmarks and generate Figure 2 from the paper:
```bash
cd experiments/02_irregular_sampling
python3 benchmark_main.py
```
_Output: `FitzHugh-Nagumo_rigorous.png` and statistical summary table._

### Experiment C.3: Optimizer Benchmark (CIFAR-10 and CHB-MIT)

#### 1) Run the sweep/grid
```bash
bash experiments/03_optimizer_compare/Gap/bench/src/run_pareto_grid.sh
```
#### 2) Aggregate results
```python
python3 experiments/03_optimizer_compare/Gap/bench/src/aggregate_results.py
```

#### 3) Make plots (Figure 4/5-style)
```python
python3 experiments/03_optimizer_compare/Gap/bench/src/plot_acc_time.py
python3 experiments/03_optimizer_compare/Gap/bench/src/plot_pareto_tradeoff.py
```

#### 4) Export LaTeX table (if used in the paper)
```python
python3 experiments/03_optimizer_compare/Gap/bench/src/export_latex_table.py
```

## Citation
If you use this code or our findings in your research, please cite the paper:

```bibtex
@article{zainal2026stabilityfirst,
  title={Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design},
  author  = {Zainal, Zayn Andre and Kavahei, Omid and Aguilar, Isabelle and Herbozo Cortez, Luis Fernando and Huang, Zhaojing},
  journal={(TBA)},
  year={2026},
  institution={School of Biomedical Engineering, University of Sydney}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Corresponding author e-mail**: Zayn Andre Zainal (andre.zainal@sydney.edu.au)
