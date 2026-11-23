"""
Simplest NP (nutrient-phytoplankton) model from:

Hodges, B.A. & Rudnick, D.L. (2004), "Simple models of steady deep maxima
in chlorophyll and biomass", Deep-Sea Research I 51, 999-1015.

This solves the Section-3 model (their Eqs. 4-5): a steady-state,
one-dimensional, two-compartment (N, P) model with diffusion only -
NO sinking term. Growth is linear in nutrient concentration and decays
exponentially with depth (light attenuation); loss of P is a constant
specific death rate that returns nitrogen directly to N.

    dN/dt = -G*N*P*exp(-K*d) + D*P + kappa * d2N/dd2
    dP/dt =  G*N*P*exp(-K*d) - D*P + kappa * d2P/dd2

    d = depth below the surface (m), d = 0 at the surface, increasing
        downward (note: the paper's own z-coordinate runs the opposite
        way, increasing upward from 0 at the surface - using depth "d"
        here is just the more intuitive, equivalent choice).

Boundary conditions:
    dN/dd = dP/dd = 0        at d = 0        (no flux through the surface)
    N = 1, P = 0              at d = L (deep) (normalized deep nutrient
                                                value; no phytoplankton at
                                                great depth)

The paper shows analytically (via S = N+P) that this system can NEVER
produce a deep phytoplankton maximum: S(d) = 1 everywhere, so a maximum
in P must coincide with a minimum in N, and because growth always
needs light AND nutrients together, the phytoplankton maximum is
forced to sit right at the surface. This script reproduces their
Figure 3.
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parameters (dimensional), matching the paper's Figure 3
# ---------------------------------------------------------------------
G_day = 20.0        # max growth rate constant, day^-1
D_day = 0.1         # specific death rate, day^-1
kappa = 1.0e-4       # turbulent diffusivity, m^2 s^-1
K = 0.05             # light attenuation coefficient, m^-1
L = 800.0            # depth of the (effectively infinite) bottom boundary, m

# convert day^-1 rates to s^-1 so everything is in SI (depth in m, time in s)
G = G_day / 86400.0
D = D_day / 86400.0


def rhs(d, y):
    """
    State vector y = [N, N', P, P'] as a function of depth d (m).
    Returns dy/dd.
    """
    N, Np, P, Pp = y
    growth_minus_death = G * N * P * np.exp(-K * d) - D * P
    dN2 = growth_minus_death / kappa      # N'' = (G N P e^{-Kd} - D P)/kappa
    dP2 = -dN2                             # P'' = -N''  (since d/dd(N+P)=const)
    return np.vstack([Np, dN2, Pp, dP2])


def bc(y0, yL):
    """
    Boundary conditions:
      at d=0 (surface): N'(0) = 0, P'(0) = 0   (no-flux surface)
      at d=L (deep):    N(L) = 1,  P(L) = 0
    """
    N0, Np0, P0, Pp0 = y0
    NL, NpL, PL, PpL = yL
    return np.array([Np0, Pp0, NL - 1.0, PL - 0.0])


def solve_np_model(n_points=400):
    d_grid = np.linspace(0.0, L, n_points)

    # Initial guess: N rising from ~0 at surface to 1 at depth,
    # P falling from ~1 at surface to 0 at depth (a simple smooth guess;
    # solve_bvp just needs something reasonable to start from).
    N_guess = 1.0 - np.exp(-d_grid / 100.0)
    P_guess = np.exp(-d_grid / 100.0)
    Np_guess = np.gradient(N_guess, d_grid)
    Pp_guess = np.gradient(P_guess, d_grid)
    y_guess = np.vstack([N_guess, Np_guess, P_guess, Pp_guess])

    sol = solve_bvp(rhs, bc, d_grid, y_guess, max_nodes=200000, tol=1e-8, verbose=0)

    if not sol.success:
        raise RuntimeError(f"solve_bvp failed to converge: {sol.message}")

    return sol


def plot_solution(sol, depth_max=250.0, outfile="np_model_no_sinking.png"):
    d_plot = np.linspace(0.0, depth_max, 400)
    N, _, P, _ = sol.sol(d_plot)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(P, -d_plot, label="Phytoplankton (P)", color="tab:green", lw=2)
    ax.plot(N, -d_plot, label="Nutrients (N)", color="tab:blue", lw=2, linestyle="--")
    ax.set_xlabel("Nutrients and Phytoplankton (nondimensional)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        "NP model without sinking (Eqs. 4-5)\n"
        f"G={G_day} day$^{{-1}}$, D={D_day} day$^{{-1}}$, "
        f"$\\kappa$={kappa:.0e} m$^2$ s$^{{-1}}$"
    )
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.show()
    return fig


if __name__ == "__main__":
    solution = solve_np_model()

    # Report key diagnostics
    d_fine = np.linspace(0.0, L, 5000)
    N_fine, _, P_fine, _ = solution.sol(d_fine)

    p_max_idx = np.argmax(P_fine)
    print("Solved the no-sinking NP model (Section 3 of Hodges & Rudnick 2004).")
    print(f"  Max phytoplankton P = {P_fine[p_max_idx]:.4f} at depth = {d_fine[p_max_idx]:.1f} m")
    print(f"  (As the paper proves analytically, this maximum must sit at the")
    print(f"   surface, d=0, since N+P=1 everywhere and growth needs both light")
    print(f"   and nutrients -- there is no deep phytoplankton maximum here.)")
    print(f"  N(0) = {N_fine[0]:.4f}, P(0) = {P_fine[0]:.4f}")
    print(f"  N(L) = {N_fine[-1]:.4f}, P(L) = {P_fine[-1]:.4f}")

    plot_solution(solution)