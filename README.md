# Stability-First Biomedical AI Should Replace Discrete-Time Defaults and Curvature-Blind Training

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

### 1. Irregular Sampling Benchmark (Appendix A.1)

_Located in: `experiments/02_irregular_sampling/`_

Demonstrates the "Discrete-Time Fallacy" by benchmarking Neural ODEs against Time-Aware LSTMs and ODE-RNNs on chaotic FitzHugh-Nagumo neuronal dynamics.
- **Key Finding**: Discrete RNNs accumulate significant drift when data is sparse or irregularly sampled, whereas Neural ODEs learn the continuous vector field.
- **Metrics**: RMSE, Phase Space Trajectory Divergence.

### 2. Edge of Stability Visualization (Appendix A.2)

_Located in: `experiments/02_edge_of_stability/`_
Investigates optimization instability in deep networks trained on biomedical data (MedMNIST).
- **Key Finding**: Modern optimizers do not converge to flat minima but "surf" the walls of high-curvature valleys ($\lambda_{max} > 2/\eta$), creating a risk of fragility in safety-critical deployments.
- **Visualization**: 3D PCA projections of the loss landscape and Hessian spectral estimation.

## **Getting Started**

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

### Experiment A.1: Irregular Sampling (FitzHugh-Nagumo)

To reproduce the time-series benchmarks and generate Figure 2 from the paper:
```bash
cd experiments/01_irregular_sampling
python3 benchmark_main.py
```
_Output: `FitzHugh-Nagumo_rigorous.png` abd statistical summary table._

### Experiment A.2: Edge of Stability (MedMNIST)

To reproduce the loss landscape visualization and Figure 3 from the paper:
```bash
cd experiments/02_edge_of_stability

# 1. Train the model (run for multiple seeds for confidence intervals)
python3 train_eos.py --seed 0

# 2. Visualize the trajectory
python3 visualize.py --checkpoint 0

# 3. Visualize multi-seed trajectory (publication figure)
python3 visualize_multi.py --seed 0
```

_Output: `figures/eos_single_0.png` showing the 3D optimization path._

## Citation
If you use this code or our findings in your research, please cite the paper:

```bibtex
@article{zainal2026neural,
  title={Neural Network Dynamics in Biomedical Applications: Reviewing the Gap Between Optimization Instability and Theory-Driven Design},
  author={Zainal, Zayn Andre and Huang, Zhaojing and Aguilar, Isabelle and Kavahei, Omid},
  journal={(TBA)},
  year={2026},
  institution={School of Biomedical Engineering, University of Sydney}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Corresponding author e-mail**: Andre Zainal (andre.zainal@sydney.edu.au)
