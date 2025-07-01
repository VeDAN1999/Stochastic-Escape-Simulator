# Stochastic-Escape-Simulator
This repository contains a numerical and analytical study of escape rates in a one-dimensional stochastic system modeled by a quartic potential as an example. The code simulates noisy trajectories escaping from a metastable well under varying noise intensities and compares:
- Numerical escape rates from stochastic simulations
- Kramers' approximation (semiclassical rate formula)
- Exact escape rates computed via a double integral formulation
In future projects, we will extend to coloured noise, as well as use resurgence theory to improve Kramers' escape formula, but this repository gives the foundational material.
# Background material
The notion of a particle overcoming a potential barrier driven by random fluctuations is ubiquituous in the physical sciences. For example, in chemical kinetics, the barrier is known as the activation energy and the escape rate is known as the chemical reaction rate. From the decay of the false vacuum, to financial markets, the study of Brownian particles subject to stochastic forcing is a highly active research area. We partition the background material into three main sections that covers the numerical escape rate, Kramers rate and finally the exact escape rate formula. We begin with Langevin dynamics. 
## Langevin dynamics
Consider a Brownian particle moving in a conservative potential, V(x), subject to random noise. This is described (heuristically) by a Langeivn equation, dx/dt = −V′(x) + √(2D) W(t).
