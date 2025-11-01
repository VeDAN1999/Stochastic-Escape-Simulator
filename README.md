# Stochastic Escape Simulator

Numerical + analytical study of escape rates for overdamped Langevin dynamics in 1D.

## Model

The SDE (Itô) is $\mathrm{d}x_t=-V'(x_t)\,\mathrm{d}t+\sqrt{2D}\,\mathrm{d}W_t$.

Time discretisation (Euler–Maruyama):

$$
x_{n+1} = x_n - V'(x_n)\,\Delta t + \sqrt{2D\,\Delta t}\,\xi_n,\qquad \xi_n\sim\mathcal N(0,1).
$$

We compare:
- Monte Carlo MFPT/escape rate from simulation;
- **Kramers’ approximation**

  $$
  \Gamma_K \approx \frac{\omega_a\,\omega_b}{2\pi}\,e^{-\Delta V/D}
  $$

  with $\omega_a=\sqrt{V''(a)}$, $\omega_b=\sqrt{|V''(b)|}$, and $\Delta V=V(b)-V(a)$;
- an **exact** double-integral MFPT formula for 1D diffusion.


## Quickstart

```bash
git clone https://github.com/VeDAN1999/Stochastic-Escape-Simulator
cd Stochastic-Escape-Simulator
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/kramers_rate.py  # prints Γ_K and plots potential


