\# Mathematical notes



\## 1. Overdamped escape model



We consider the one-dimensional overdamped Langevin equation



```math

dx\_t = -V'(x\_t)\\,dt + \\sqrt{2D}\\,dW\_t.

```



Here:



\- `V(x)` is the potential

\- `D` is the diffusion strength

\- `W\_t` is standard Brownian motion



For a metastable potential well, we study the mean time required for a trajectory started near the local minimum to cross an absorbing threshold beyond the barrier.



\---



\## 2. Kramers' rate



For small noise, the escape rate is approximated by Kramers' formula



```math

\\Gamma\_K

=

\\frac{1}{2\\pi}

\\sqrt{|V''(x\_a)V''(x\_b)|}

\\exp\\left(-\\frac{V(x\_b)-V(x\_a)}{D}\\right),

```



where:



\- `x\_a` is the metastable minimum

\- `x\_b` is the barrier top



This gives the expected Arrhenius behaviour in the weak-noise regime.



\---



\## 3. Exact one-dimensional rate



In one dimension, the mean first-passage time from `x\_0` to an absorbing point `x\_abs` is



```math

T(x\_0)

=

\\int\_{x\_0}^{x\_{\\mathrm{abs}}}

\\frac{e^{V(y)/D}}{D}

\\left(

\\int\_{-\\infty}^{y} e^{-V(z)/D}\\,dz

\\right)\\,dy.

```



The exact rate used in the code is then



```math

\\Gamma\_{\\mathrm{exact}} = \\frac{1}{T(x\_0)}.

```



\---



\## 4. Numerical simulation



The code uses Euler–Maruyama time stepping:



```math

x\_{n+1} = x\_n - V'(x\_n)\\Delta t + \\sqrt{2D\\Delta t}\\,Z\_n,

```



with independent standard normal variables `Z\_n`.



A first-passage time is recorded when the trajectory first crosses the absorbing threshold. To reduce discretisation bias, the crossing time is refined by linear interpolation within the final step.



\---



\## 5. Coloured-noise extension



An optional coloured-noise version is included by introducing an Ornstein–Uhlenbeck auxiliary process with correlation time `tau\_c`. This generates exponentially decaying temporal correlations and provides a simple extension beyond white noise.



\---



\## 6. Purpose of the repository



The aim of the repository is not to be a large general-purpose package, but rather a clean and reproducible demonstration of:



\- stochastic simulation

\- first-passage methods

\- asymptotic escape-rate theory

\- comparison between theory and numerics

