# Stochastic Escape Simulator

Numerical + analytical study of escape rates for overdamped Langevin dynamics in 1D.

## Model

The Stochastic differential equation (SDE) is $\mathrm{d}x_t=-V'(x_t) \mathrm{d}t+\sqrt{2D}\mathrm{d}W_t$ where $\mathmrm{d}W_t$ is a Wiener process drawn from a normal distribution with mean zero. $D$ is the noise strength o or diffusion coefficient. This is the formal version of the overdamped Langevin equation.  

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

with $\omega_{a}=\sqrt{V''(a)}$, $\omega_{b}=\sqrt{\lvert V''(b) \rvert}$, and $\Delta V =V(b)-V(a)$;
- an **exact** double-integral MFPT formula for 1D diffusion.


## Quickstart

```bash
git clone https://github.com/VeDAN1999/Stochastic-Escape-Simulator
cd Stochastic-Escape-Simulator
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/kramers_rate.py  # prints Γ_K and plots potential


