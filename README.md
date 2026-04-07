# Stochastic Escape Simulator

Numerical and analytical study of escape rates for overdamped Langevin dynamics in one dimension.

This repository compares three approaches to metastable escape in a tilted quartic potential:

- **Numerical simulation** using Euler–Maruyama time stepping and first-passage times
- **Kramers' rate** in the weak-noise regime
- **Exact 1D rate** computed from the mean first-passage-time integral

An optional extension includes **exponentially correlated coloured noise** via an Ornstein–Uhlenbeck auxiliary process.

---

## Features

- Simulation of overdamped stochastic escape in a metastable potential
- Comparison of numerical, exact, and asymptotic escape rates
- Arrhenius-style plots of `log(rate)` against `1/D`
- Optional coloured-noise switch
- Numba acceleration for faster Monte Carlo runs

---

## Repository structure

```text
.
├─ docs/
│  └─ maths.md
├─ images/
│  └─ arrhenius_example.png
├─ src/
│  └─ stochastic_escape_simulator.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt

