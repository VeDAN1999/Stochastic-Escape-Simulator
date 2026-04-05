import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from numba import njit
from matplotlib.lines import Line2D
import scienceplots  



# ============================================================
# Potential, force, curvature
# ============================================================

def potential(x: float) -> float:
    return x**4 - 4*x**3 + 4*x**2 - 0.6*x + 1.6


def force(x: float) -> float:
    # F = -V'(x)
    return -4*x**3 + 12*x**2 - 8*x + 0.6


def vpp(x: float) -> float:
    return 12*x**2 - 24*x + 8


# ============================================================
# Problem parameters
# ============================================================

# Metastable minimum and barrier location for this quartic
a = 0.0857
b = 0.8464

# Start at the well minimum if comparing to Kramers / exact MFPT
x0 = a

# Absorbing point chosen slightly beyond the barrier
x_absorb = 1.05

# Barrier height and Kramers prefactor
E_b = potential(b) - potential(a)
omega_k = (1.0 / (2.0 * np.pi)) * np.sqrt(abs(vpp(a)) * abs(vpp(b)))

# Simulation parameters
dt = 0.01
Nsteps = 50000
N_particles = 500

# Keep this modest at first so the exact integral is not too slow
D_vals = np.linspace(0.12, 2.5, 80)

# Noise mode:
#   0 = white noise
#   1 = exponentially correlated noise (OU auxiliary variable)
noise_mode = 0

# Correlation time for coloured noise
tau_c = 0.05


# ============================================================
# Numba kernels
# ============================================================

@njit
def force_numba(x):
    return -4.0*x**3 + 12.0*x**2 - 8.0*x + 0.6


@njit
def simulate_escape_statistics(
    N_particles,
    Nsteps,
    dt,
    D_vals,
    x0,
    x_absorb,
    noise_mode,
    tau_c
):
    """
    Returns:
        mean_times[j]      = mean first-passage time for D_vals[j]
        escape_fraction[j] = fraction escaped before final time
    """
    nD = len(D_vals)
    mean_times = np.full(nD, np.nan)
    escape_fraction = np.zeros(nD)

    for j in range(nD):
        D = D_vals[j]

        X = np.full(N_particles, x0)
        alive = np.ones(N_particles, dtype=np.uint8)
        hit_times = np.zeros(N_particles)

        # For coloured noise, use auxiliary OU variable eta
        if noise_mode == 1:
            alpha = math.exp(-dt / tau_c)
            # Stationary variance Var(eta) = 1 / (2 tau_c)
            sigma_eta = math.sqrt((1.0 - alpha * alpha) / (2.0 * tau_c))
            eta = np.random.normal(0.0, 1.0 / math.sqrt(2.0 * tau_c), N_particles)

        n_alive = N_particles

        for n in range(1, Nsteps + 1):
            t_prev = (n - 1) * dt

            for i in range(N_particles):
                if alive[i] == 0:
                    continue

                x_old = X[i]

                if noise_mode == 0:
                    # White noise: Euler-Maruyama
                    x_new = x_old + dt * force_numba(x_old) + math.sqrt(2.0 * D * dt) * np.random.normal()

                else:
                    # Exponentially correlated noise via OU auxiliary process:
                    #   dx/dt = F(x) + sqrt(2D) eta(t)
                    #   deta = -(1/tau_c) eta dt + (1/tau_c) dW
                    eta[i] = alpha * eta[i] + sigma_eta * np.random.normal()
                    x_new = x_old + dt * force_numba(x_old) + math.sqrt(2.0 * D) * eta[i] * dt

                X[i] = x_new

                # First-passage detection with linear interpolation
                if x_old <= x_absorb and x_new > x_absorb:
                    denom = x_new - x_old
                    if denom > 0.0:
                        frac = (x_absorb - x_old) / denom
                        frac = min(max(frac, 0.0), 1.0)
                    else:
                        frac = 1.0

                    hit_times[i] = t_prev + frac * dt
                    alive[i] = 0
                    n_alive -= 1

            if n_alive == 0:
                break

        n_escaped = N_particles - n_alive
        escape_fraction[j] = n_escaped / N_particles

        if n_escaped > 0:
            mean_times[j] = np.sum(hit_times) / n_escaped

    return mean_times, escape_fraction


# ============================================================
# Exact MFPT / exact rate
# ============================================================

def exact_rate(D, x_start, x_absorb):
    """
    Overdamped 1D MFPT from x_start to x_absorb:
        T(x_start) = ∫_{x_start}^{x_absorb} dy [ e^{V(y)/D} / D ] ∫_{-∞}^{y} dz e^{-V(z)/D}

    We evaluate the product as exp((V(y) - V(z))/D) / D for better numerical stability.
    """
    val, _ = dblquad(
        lambda z, y: np.exp((potential(y) - potential(z)) / D) / D,
        x_start,
        x_absorb,
        lambda y: -np.inf,
        lambda y: y
    )
    return 1.0 / val


# ============================================================
# Run numerics
# ============================================================

mean_times, escape_fraction = simulate_escape_statistics(
    N_particles=N_particles,
    Nsteps=Nsteps,
    dt=dt,
    D_vals=D_vals,
    x0=x0,
    x_absorb=x_absorb,
    noise_mode=noise_mode,
    tau_c=tau_c
)

# Mark strongly censored estimates as NaN
min_escape_fraction_for_reporting = 0.95
numerical_rates = np.full_like(mean_times, np.nan)

for j in range(len(D_vals)):
    if not np.isnan(mean_times[j]) and escape_fraction[j] >= min_escape_fraction_for_reporting:
        numerical_rates[j] = 1.0 / mean_times[j]

# Kramers rate
kramers_rates = omega_k * np.exp(-E_b / D_vals)

# Exact rates
exact_rates = np.array([exact_rate(D, x0, x_absorb) for D in D_vals])


# ============================================================
# Plotting
# ============================================================

invD = 1.0 / D_vals
log_kramers = np.log(kramers_rates)
log_exact = np.log(exact_rates)

mask_num = np.isfinite(numerical_rates)
invD_num = invD[mask_num]
log_num = np.log(numerical_rates[mask_num])

marker_idxs_exact = np.linspace(0, len(D_vals) - 1, 16, dtype=int)
marker_idxs_num = np.linspace(0, len(invD_num) - 1, min(16, len(invD_num)), dtype=int) if len(invD_num) > 0 else np.array([], dtype=int)

title = "Kramers vs exact and numerical escape rates"
if noise_mode == 1:
    title += f" (coloured noise, tau_c = {tau_c:g})"

custom_lines = [
    Line2D([0], [0], color="green", lw=2, marker="s", label="Exact rate"),
    Line2D([0], [0], color="red", lw=2, linestyle="--", marker="o", label="Kramers rate"),
    Line2D([0], [0], color="blue", marker="^", linestyle="None", label="Numerical rate"),
    Line2D([0], [0], color="black", lw=1, linestyle="--", label=r"$1/E_b$")
]

with plt.style.context(["science", "no-latex", "ieee", "high-vis"]):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # Exact
    ax.plot(invD, log_exact, color="green", linestyle="-", linewidth=1.5)
    ax.plot(invD[marker_idxs_exact], log_exact[marker_idxs_exact], "s", color="green", markersize=3)

    # Kramers
    ax.plot(invD, log_kramers, color="red", linestyle="--", linewidth=1.5)
    ax.plot(invD[marker_idxs_exact], log_kramers[marker_idxs_exact], "o", color="red", markersize=3)

    # Numerical
    if len(invD_num) > 0:
        ax.plot(invD_num, log_num, "^", color="blue", markersize=3, linestyle="None")

    # Vertical barrier scale
    ax.axvline(x=1.0 / E_b, color="black", linestyle="--", linewidth=1.0)

    # Semiclassical / non-semiclassical shading
    ax.axvspan(1.0, 1.0 / E_b, color="gray", alpha=0.2)

    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$\log \Gamma$")
    ax.set_title(title, fontsize=10)

    ax.set_xlim(1.0, max(invD))
    y_min = min(np.nanmin(log_exact), np.nanmin(log_kramers), np.nanmin(log_num) if len(log_num) > 0 else np.nanmin(log_exact))
    y_max = max(np.nanmax(log_exact), np.nanmax(log_kramers), np.nanmax(log_num) if len(log_num) > 0 else np.nanmax(log_exact))
    ax.set_ylim(y_min - 0.4, y_max + 0.4)

    ax.text(
        1.0 / E_b + 0.15,
        y_min + 0.2,
        r"$D \gtrsim E_b$",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2)
    )

    ax.legend(
        handles=custom_lines,
        fontsize=8.5,
        loc="upper right",
        frameon=True,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.4
    )

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ============================================================
# Diagnostic printout
# ============================================================

print("Barrier height E_b =", E_b)
print("Kramers prefactor =", omega_k)
print("Minimum escape fraction across D grid =", np.min(escape_fraction))
