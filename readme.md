# Differentiable Non-Linear Optimization as a Layer in Neural Network Architectures

![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Thesis](https://img.shields.io/badge/Thesis-KU%20Leuven-005b99)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./License)

Code and simulations for my Master's thesis, the final work of the **Advanced Master's in Artificial Intelligence** (Engineering and Computer Science option) at **KU Leuven**, academic year 2022–2023. Supervisor: Prof. Yves Moreau. Assistant-supervisor: Antoine Passemiers.

📄 **Full thesis:** [Marco Salmistraro - Thesis.pdf](./Marco%20Salmistraro%20-%20Thesis.pdf)

---

## Overview

A standard neural network layer is an explicit function of its input. A **constrained optimization problem** can also be used as a layer: its output is the *solution* of an optimization task whose parameters are produced by the previous layer. This lets a network encode constraints and variable dependencies that fully-connected layers struggle to represent, and acts as a form of implicit regularization on the output. The difficulty is that training requires gradients to flow back through the solver.

The work is inspired by **OptNet** (Amos & Kolter, 2017), which provides closed-form back-propagation through **quadratic programs (QPs)**. This project extends the idea to **general non-linear programs (NLPs)** and provides a PyTorch implementation that can be adapted to arbitrary parameterizations of non-linear constrained problems.

## Approach

To demonstrate the method, the repo implements an **autoencoder whose decoder is a parameterized non-linear constrained optimization problem**.

![Model architecture](https://github.com/marcosalmistraro/differentiable_optimization/raw/main/imgs/FIG_5.png)

- The **encoder** is a multi-layer neural network that maps the input `x` to a latent vector `ẋ`.
- The values in `ẋ` are the **learned parameters** of the optimization problem in the decoder.
- The **decoder** solves that problem with a non-linear solver (SLSQP), starting from `x`, to produce the reconstruction `x̂`.
- The loss is the **MSE** between `x` and `x̂`.

### Decoder problem

The test problem is a parameterized version of the **Rosenbrock function** with one cubic and one linear constraint:

```
minimize    (1 − x)² + 100 (y − x²)²
subject to  (x − 1)³ − y + 1 ≤ 0
            x + y − 2 ≤ 0
```

Learnable parameters replace constants in the objective and/or the constraints, for example `(α − x)² + β (y − x²)²` with a constraint `γ (x + y) − 2 ≤ 0`.

### Gradient back-propagation

Rather than deriving a closed-form solution for every non-linear case, gradients are obtained from a **local approximation around the solution `x̂`**:

- the **objective** is approximated with a **quadratic** (second-order Taylor expansion)
- the **constraints** are approximated with **linear** (first-order Taylor) terms

This produces OptNet-style parameters (`Q, q, G, h, A, b`). The optimal Lagrange multipliers (`λ*`, `ν*`) are then computed from the KKT conditions at `x̂`, and the closed-form OptNet expressions give the backward gradients. These are wired into the PyTorch graph through custom backward methods: a vector-Jacobian product for the quadratic approximation and a (least-squares) linear system for the linear constraint approximations.

![PyTorch graph structure](https://github.com/marcosalmistraro/differentiable_optimization/raw/main/imgs/FIG_8.png)

The differential matrix in the OptNet system can be singular for these problems. It is made invertible by adding a very small value to its diagonal.

## Results

Training data was sampled uniformly from [0, 1] × [0, 1]. All runs used Adam with a learning rate of `1e-5`.

| Run | Learned parameters | Data | Iterations | Outcome |
|---|---|---|---|---|
| 1 | α, β (objective) | 1 point | 3000 | MSE decreases steadily, converging to a suitable latent representation |
| 2 | α, β (objective), γ (constraint) | 1 point | 3000 | Parameters learned, with a fast, roughly exponential drop in MSE |
| 3 | α, β, γ | 50 points | 6000 | Slower convergence, as expected, but steady improvement |
| 4 | α, β, γ, δ, ε (objective + both constraints, non-linear injection) | 50 points | 6000 | Converges, but to a higher error (MSE plateaus at ≥ 1) |

**Takeaways**

- Gradients propagate correctly through the non-linear decoder for parameters in both the objective and the constraints, so the model learns.
- More complex parameterizations lead to a higher training error.
- Convergence is **not always reproducible**: runs with the same data can differ, and the 5-parameter case showed gradient issues and instability. The thesis discusses likely causes (weight initialization, activation functions, and the SLSQP solver itself).
- The goal was to demonstrate that general NLPs can be embedded in end-to-end networks, not to build a model for a specific application. The 1-point runs are proofs of feasibility and would not generalize to unseen data.

## Limitations and future work

- Gradients are **approximations** (local quadratic and linear). A closed-form derivation for general non-linear problems would remove this.
- No sparse matrix operations are used, which could reduce the cost on larger datasets.
- Backward gradient computation could be parallelized for higher-dimensional data.
- The local solver can get stuck in local minima. Combining it with a global optimizer is an open direction.

## Repository Structure

```
differentiable_optimization/
├── NL_problem/                        # Non-linear problem implementation and simulations
│   ├── diffcp.py                      # Shared DiffCP autograd Function (used by every run)
│   ├── first_run/                     # Run 1: 1 data point, parameters α, β
│   │   ├── AE_1point_2param.py
│   │   └── decoder_functions.py
│   ├── second_run/                    # Run 2: 1 data point, parameters α, β, γ
│   │   ├── AE_1point_3param.py
│   │   └── decoder_functions.py
│   ├── third_run/                     # Run 3: 50 data points, parameters α, β, γ
│   │   ├── AE_50points_3param.py
│   │   └── decoder_functions.py
│   └── fourth_run/                    # Run 4: 50 data points, 5 parameters
│       ├── AE_50points_5param.py
│       └── decoder_functions.py
├── imgs/                              # Figures used in this README
├── Marco Salmistraro - Thesis.pdf     # Full thesis text
├── readme.md
└── .gitignore
```

`NL_problem/` holds the shared solver module plus one folder per experiment (`first_run`, `second_run`, `third_run`, `fourth_run`), matching the four runs in the [Results](#results) table.

- **`diffcp.py`**: the core of the project, shared by all runs. It defines `DiffCP`, a custom `torch.autograd.Function` that solves the non-linear constrained problem in the forward pass and back-propagates through it. The implementation is identical across runs, so it lives in one place rather than being duplicated per run.

Each run folder then contains:

- **`decoder_functions.py`**: the run-specific problem definition — the Rosenbrock objective and the constraints for that run. It re-exports `DiffCP` from `../diffcp.py`, so existing `decoder_functions.DiffCP` references keep working.
- **A run script** (`AE_<points>_<params>.py`): defines the autoencoder (`AE`), builds the dataset, trains the model and plots the loss curve. Tunable settings (network widths, number of samples, iterations, learning rate, seed) are named constants at the top of the file.

### How the code works

- **Encoder:** a small network (`Linear(2, 16)` → `Tanh` → `Linear(16, 4)`, the same in every run) whose outputs are the learned parameters of the decoder problem.
- **Forward pass (`DiffCP.forward`):** solves the problem with SciPy's SLSQP, using the input as the starting point. It then builds the local quadratic and linear approximations at the solution and recovers the Lagrange multipliers by solving the KKT system with least squares.
- **Backward pass (`DiffCP.backward`):** applies the OptNet gradient expressions to the approximated problem. The gradient for the objective parameters is obtained with a vector-Jacobian product. A flag, `USE_VJP_BASED_BACKWARD_IMPLEMENTATION`, switches to a slower Jacobian-based alternative.
- **Training:** Adam (`lr=1e-5`), MSE loss, and a dataset sampled uniformly from [0, 1]² with a fixed seed (`42`).

## Getting Started

**Requirements**

- Python 3.x
- [PyTorch](https://pytorch.org/)
- NumPy
- SciPy (for the SLSQP solver)
- Matplotlib (loss plots)
- tqdm (progress bar)

```bash
pip install torch numpy scipy matplotlib tqdm
```

**Run an experiment**

The run scripts import `decoder_functions` as a local module, so run each one from inside its own run folder:

```bash
git clone https://github.com/marcosalmistraro/differentiable_optimization.git
cd differentiable_optimization/NL_problem/first_run
python AE_1point_2param.py
```

The four experiments:

| Run | Folder | Script | Setup |
|---|---|---|---|
| 1 | `first_run` | `AE_1point_2param.py` | 1 data point, 2 parameters (α, β) |
| 2 | `second_run` | `AE_1point_3param.py` | 1 data point, 3 parameters (α, β, γ) |
| 3 | `third_run` | `AE_50points_3param.py` | 50 data points, 3 parameters (α, β, γ) |
| 4 | `fourth_run` | `AE_50points_5param.py` | 50 data points, 5 parameters (α, β, γ, δ, ε) |

The script trains the autoencoder and then opens a plot of the loss over the training iterations. During training, the gradient estimate for the objective parameters is logged at every iteration (via the standard `logging` module, at `DEBUG` level), which is useful for tracking convergence. To see it, enable debug logging before running, e.g.:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Using it on your own problem

`DiffCP` is called through `DiffCP.apply(...)` with the problem defined by plain Python functions:

```python
DiffCP.apply(obj_param, obj,        # objective parameters and function
             eq_param, eqs,         # equality constraint parameters and functions
             ineq_param, ineqs,     # inequality constraint parameters and functions
             x_input)               # starting point for the SLSQP solver
```

`DiffCP` lives in `NL_problem/diffcp.py`; import it directly with `from diffcp import DiffCP`, or via a run's `decoder_functions` module, which re-exports it.

- The objective is a function `obj(x, param)` that returns a scalar tensor.
- Constraints are lists of functions in the SciPy convention (`ineq` constraints must be ≥ 0).
- The parameters are tensors produced by your encoder network.

See the `rosenbrock` and `ineq_constraint_*` functions in `decoder_functions.py` for examples.

## Reference

- B. Amos and J. Z. Kolter. *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML, 2017.

## Author

**Marco Salmistraro**  
Advanced MSc in Artificial Intelligence, KU Leuven  
GitHub: [@marcosalmistraro](https://github.com/marcosalmistraro) · [LinkedIn](https://www.linkedin.com/in/marco-salmistraro-259078105/)

## License

This project is licensed under the MIT License. See the [License](./License) file for details.
