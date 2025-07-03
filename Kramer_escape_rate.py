mport numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from numba import njit
import math
import scienceplots

(''' Now optimised the escape rate code''')

# --- A 1D potential and its curvature. Working with a tilted quartic for easy demonstration but can be changed as you please ---
def potential(x):
    return x**4 - 4*x**3 + 4*x**2 - 0.6*x + 1.6

def vpprime(x):
    return 12*x**2 - 24*x + 8

# --- Potential Constants and parameters.  ---
q0 = 0.0857  # Starting point (local minimum)
a = 0.086 # Metastable well
b = 0.846 # The barrier
E_b = potential(b) - potential(a)  # Barrier height

w = (1 / (2*np.pi)) * np.sqrt(abs(vpprime(a)) * abs(vpprime(b)))  # Standard Kramers' Prefactor
dt = 0.01 # Time step. 0.01 is a standard and (usually) reliable choice but can be made smaller.
sdt = math.sqrt(dt) # Euler-Maruyama scheme and the Wiener process -- setting up the sqrt(2D dt) factor.
N = 500  # Number of particles 
Nsteps = 50000 # Total number of Time steps, i.e. full time interval.
mu, sigma = 0, 1  # Wiener process parameters -- zero mean and variance unity.

# Range of noise strengths. 
D_vals = np.linspace(0.1, 4, 300)

# --- Numba-accelerated functions for performance optimisation ---
# --- Conservative force in the Langevin equation: F = - V'(x) ---
@njit
def force_numba(x):
    return -4*x**3 + 12*x**2 - 8*x + 0.6
# --- The Main code ---
# --- x_i+1 = x+i + force dt + sqrt(2 D W_t) --- Euler Maruyama algorithm
@njit
def simulate_escape_times(N, Nsteps, dt, sdt, D_vals, q0):
    escape_time = np.zeros((len(D_vals), N))
    for idx_D in range(len(D_vals)):
        D_val = D_vals[idx_D]
        X = np.full(N, q0)
        escaped = np.zeros(N)

        for t in range(Nsteps):
            noise = sdt * np.sqrt(2 * D_val) * np.random.normal(0, 1, N)
            X += dt * force_numba(X) + noise
         # --- Define a point where you can be confident the particle has fully surpassed the barrier. 1.05 is chosen arbitrarily here, but can be fine-tuned.
            for i in range(N):
                if X[i] > 1.05 and escaped[i] == 0:
                    escape_time[idx_D, i] = t * dt
                    escaped[i] = 1
    return escape_time

# --- Run simulation and store all in one big array---
escape_time_array = simulate_escape_times(N, Nsteps, dt, sdt, D_vals, q0)

# Remove zeros (particles that didn't escape) and compute mean escape times
mean_times = []
for times in escape_time_array:
    filtered = times[times > 0]
    mean = np.mean(filtered) if len(filtered) > 0 else np.nan
    mean_times.append(mean)

mean_times = np.array(mean_times)
numerical_rates = np.reciprocal(mean_times)

# --- Kramers' rate ---
kramers_rates = w * np.exp(-E_b / D_vals)

# --- Exact rate from double integral assuming delta function initial condition---
def integrand(x, y, D):
    return (1 / D) * np.exp(potential(y) / D) * np.exp(-potential(x) / D)

exact_rates = []
for D in D_vals:
    val, _ = dblquad(lambda x, y: integrand(x, y, D), 0, 1.05, lambda x: -np.inf, lambda x: x)
    exact_rates.append(1 / val)
   
lst = [numerical_rates, kramers_rates,exact_rates]

# --- Plotting: This is completely up to your taste. I provided my way of presenting it for completeness. I am a fan of the science package and recommend you check it out if not already.---
colormap = np.array(['blue','green','orange','r'])
k = len(colormap)
pparam2 = dict(xlabel = r"$1/D$", ylabel = r"$\log \Gamma$")
invD = 1/D_vals
log_kramers = np.log(kramers_rates)

# Now plot the markers only at those visually even x-locations

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color='green', lw=2, marker='s', label="Exact rate"),
    Line2D([0], [0], color='red', lw=2, linestyle='--', marker='o', label="Kramers' rate"),
    Line2D([0], [0], color='blue', marker='^', linestyle='None', label="Numerical rate"),
    Line2D([0], [0], color='black', lw=1, linestyle='--', label=r"$1/E_b$")
]

with plt.style.context(["science","no-latex","ieee","high-vis"]):
    # --- Want to plot standard arrhenius plot: 1/D against log(Rate). Should get a straight line for Kramers' with intercept the constant w.
    # --- Inverse temperature axis ---
    invD = 1 / D_vals

# --- Precompute logs ---
    log_kramers = np.log(kramers_rates)
    log_exact = np.log(exact_rates)
    log_numerical = np.log(numerical_rates)

# --- Sparse marker selection (equispaced in x) ---
    marker_idxs = np.linspace(0, len(D_vals) - 1, 20, dtype=int)

# --- Begin the Plot ---
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Exact rate: green solid line with occasional square markers
    ax.plot(invD, log_exact, color='green', linestyle='-', linewidth=1.5, label="Exact rate")
    ax.plot(invD[marker_idxs], log_exact[marker_idxs], 's', color='green', markersize=3)

# Kramers: red dashed line with occasional circle markers
    ax.plot(invD, log_kramers, color='red', linestyle='--', linewidth=1.5, label="Kramers' rate")
    ax.plot(invD[marker_idxs], log_kramers[marker_idxs], 'o', color='red', markersize=3)

# Numerical: blue discrete triangle markers only
    ax.plot(invD, log_numerical, '^', color='blue', markersize=3, linestyle='None', label="Numerical rate")

# Barrier height vertical line
    ax.axvline(x=1/E_b, color='black', linestyle='--', linewidth=1, label=r"$1/E_b$")

# --- Axes and Labels ---
    ax.set_xlabel(r"$1/D$", fontsize=10)
    ax.set_ylabel(r"$\log \Gamma$", fontsize=10)
    ax.set_ylim(np.nanmin(log_numerical) - 0.5, np.nanmax(log_kramers) + 0.5)
    ax.text(1.57, -2, r"$D \gtrsim E_b$",
        color="black", fontsize=10,
        horizontalalignment="center", verticalalignment="center",
        bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

# --- Legend ---
    ax.legend(handles=custom_lines, fontsize=9, loc="upper right", frameon = True, handlelength=2.5,
    borderpad=0.5,
    labelspacing=0.4)

# --- Title ---
    ax.set_title(r"Kramers' formula versus exact and numerical rates", fontsize=10)
    ax.set_xlim(1, 8) # Focus more on the arrhenius tail
    ax.tick_params(labelsize=10)
    ax.axvspan(1, 1/E_b, color='gray', alpha=0.2, label="Beyond semiclassical regime")

# Optional and a bit of customisation I am a fan of: add light grid
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.4)

# --- Try and avoid wasting space ---
  plt.tight_layout()
# plt.savefig("figures/kramers_escape_rates.pdf", dpi=600)
  plt.show()

