# Stochastic Escape Simulator

Numerical + analytical study of escape rates for overdamped Langevin dynamics in one dimension. Can be extended to higher dimensions/coloured noise (deferred for later) and works for many potentials $V(x)$.

## Model

The stochastic differential equation (SDE) is

```math
\mathrm{d}x_t=-V'(x_t) \, \mathrm{d}t + \sqrt{2D} \, \mathrm{d}W_t, 
```

where $W_t$ is standard Brownian motion and $D$ is the diffusion coefficient.

Time discretisation (Euler–Maruyama):

```math
x_{n+1}=x_n - V'(x_n) \Delta t + \sqrt{2D \Delta t} \, \xi_n,\quad \xi_n\sim\mathcal N(0,1).
```

We compare:
- Monte Carlo MFPT/escape rate from simulation.
- **Kramers’ approximation:**
```math
\Gamma_K \approx \frac{\omega_a \omega_b}{2\pi} \exp(-\Delta V/D)
```
where $\omega_i$ denotes the curvature at the minimum and barrier.
- Optionally: An **exact** double-integral MFPT formula for 1D diffusion.




## Quickstart

```bash
git clone https://github.com/VeDAN1999/Stochastic-Escape-Simulator
cd Stochastic-Escape-Simulator
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/kramers_rate.py  # prints Γ_K and plots potential


