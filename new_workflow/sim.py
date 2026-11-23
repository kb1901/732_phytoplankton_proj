import os
import sys
import runpy
from types import SimpleNamespace
from collections import namedtuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)

# days: (frames,), depth: (NZ,) in m (negative down), N and P: (frames, NZ)
Result = namedtuple("Result", "days depth N P")


def load_params(params_file, **overrides):
    """Read a parameter file, then apply overrides, e.g. run(SINKING_ENABLED=True)."""
    path = next((c for c in (params_file, os.path.join(HERE, params_file)) if os.path.exists(c)), None)
    if path is None: raise FileNotFoundError(f"{params_file} not found (looked here and in {HERE})")
    p = SimpleNamespace(**{k: v for k, v in runpy.run_path(path).items() if not k.startswith("_")})
    for k, v in overrides.items():
        if not hasattr(p, k): raise KeyError(f"unknown parameter {k}")
        setattr(p, k, v)
    return p


def run(params_file="params.py", save_as=None, **overrides):
    """Run the model with the chosen parameter file; returns the data in memory.
    save_as='data/name' also writes data/name.npz."""
    p = load_params(params_file, **overrides)
    dt = p.DT_SECONDS
    steps_per_frame = max(1, round(p.SAVE_EVERY_DAYS * 86400 / dt))     # substeps between snapshots
    n_frames = int(round(p.T_END_DAYS / p.SAVE_EVERY_DAYS)) + 1         # snapshots, including t = 0

    if p.N_BOT_BC not in ("neumann", "dirichlet"): raise ValueError("N_BOT_BC must be 'neumann' or 'dirichlet'")
    neumann = p.N_BOT_BC == "neumann"

    enabled = {"growth": p.GROWTH_ENABLED, "sinking": p.SINKING_ENABLED, "mixing": p.MIXING_ENABLED}
    for name, on in enabled.items():
        total = sum(f for n, f in p.SPLITTING if n == name)
        if on and abs(total - 1) > 1e-12: raise ValueError(f"fractions of '{name}' in SPLITTING sum to {total}, not 1")
    if any(n not in enabled for n, _ in p.SPLITTING): raise ValueError("SPLITTING names must be growth, sinking, mixing")

    # --- Grid ---
    depth = np.linspace(-p.DEPTH, 0, p.NZ)       # 0 is the surface, -DEPTH is the bottom
    dz = depth[1] - depth[0]
    light_unshaded = np.exp(p.LIGHT_ATTENUATION * depth)

    # --- Mixing matrices (Crank-Nicolson), built once for each stage length used in SPLITTING ---
    mixing_cache = {}
    def mixing_solvers(h):
        if h not in mixing_cache:
            g = p.KAPPA * h / (2 * dz ** 2)
            main_diag, off_diag = (1 + 2 * g) * np.ones(p.NZ), -g * np.ones(p.NZ - 1)
            A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="lil")      # bottom Dirichlet
            A[0, 0], A[0, 1] = 1.0, 0.0
            A[-1, -2] = -2 * g
            B = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="lil")      # bottom Neumann
            B[0, 1] = -2 * g
            B[-1, -2] = -2 * g
            mixing_cache[h] = (g, splu(A.tocsc()), splu(B.tocsc()))
        return mixing_cache[h]

    # --- Processes ---
    N_bot_value = p.N_BOT_VALUE if callable(p.N_BOT_VALUE) else (lambda t: p.N_BOT_VALUE)    # number or function of t

    def apply_bottom_bc(N, P, t):
        if neumann: P[0] = p.P_BOT_VALUE
        else: N[0], P[0] = N_bot_value(t), p.P_BOT_VALUE
        return N, P

    def apply_growth(N, P, h, t):
        """One RK4 step of nutrient uptake / phytoplankton growth and mortality; t = time at stage midpoint."""
        def rates(N, P):
            light = np.exp((p.LIGHT_ATTENUATION + p.SHADING_COEFF * P) * depth) if p.SHADING_ENABLED else light_unshaded
            growth = p.GROWTH_RATE * N * P * light * p.LIGHT_FORCING(t) - p.MORTALITY_RATE * p.MORTALITY_FORCING(t) * P
            return -growth, growth
        dN1, dP1 = rates(N, P)
        dN2, dP2 = rates(N + h / 2 * dN1, P + h / 2 * dP1)
        dN3, dP3 = rates(N + h / 2 * dN2, P + h / 2 * dP2)
        dN4, dP4 = rates(N + h * dN3, P + h * dP3)
        return (N + h / 6 * (dN1 + 2 * dN2 + 2 * dN3 + dN4),
                P + h / 6 * (dP1 + 2 * dP2 + 2 * dP3 + dP4))

    def apply_sinking(P, h):
        """Sinking: central differences in space (second order), Heun in time (second order), conservative."""
        def rate(P):
            F = p.SINK_SPEED * (P[:-1] + P[1:]) / 2          # flux through the face between nodes i and i+1
            r = np.zeros_like(P)
            r[1:-1] = -(F[1:] - F[:-1]) / dz
            r[-1] = 2 * F[-1] / dz                           # top node is a half cell; no flux through the surface
            return r
        k1 = rate(P)
        return P + h / 2 * (k1 + rate(P + h * k1))

    def apply_mixing(N, P, h, t0):
        """One implicit (Crank-Nicolson) mixing step over [t0, t0+h]; the surface N flux enters here."""
        g, solve_dirichlet, solve_neumann = mixing_solvers(h)
        top_flux = h * (p.SURFACE_FLUX(t0) + p.SURFACE_FLUX(t0 + h)) / dz

        def mix(profile, flux, neumann):
            rhs = np.empty_like(profile)
            rhs[1:-1] = g * profile[:-2] + (1 - 2 * g) * profile[1:-1] + g * profile[2:]
            rhs[-1] = 2 * g * profile[-2] + (1 - 2 * g) * profile[-1] + flux
            if neumann:
                rhs[0] = (1 - 2 * g) * profile[0] + 2 * g * profile[1]
                return solve_neumann.solve(rhs)
            rhs[0] = profile[0]
            return solve_dirichlet.solve(rhs)

        return mix(N, top_flux, neumann), mix(P, 0.0, False)

    # --- One time step: run the stages in SPLITTING; each is followed by the bottom BC ---
    def step(N, P, t):
        used = {"growth": 0.0, "sinking": 0.0, "mixing": 0.0}      # each process keeps its own clock
        for name, frac in p.SPLITTING:
            if not enabled[name]: continue
            h, t0 = frac * dt, t + used[name] * dt
            if name == "growth": N, P = apply_growth(N, P, h, t0 + h / 2)
            elif name == "sinking": P = apply_sinking(P, h)
            else: N, P = apply_mixing(N, P, h, t0)
            used[name] += frac
            N, P = apply_bottom_bc(N, P, t + used[name] * dt)
        return N, P

    # --- Time loop ---
    N = np.full(p.NZ, p.N_INIT)
    P = np.full(p.NZ, p.P_INIT)
    N[0], P[0] = N_bot_value(0.0), p.P_BOT_VALUE
    N_history = np.zeros((n_frames, p.NZ))
    P_history = np.zeros((n_frames, p.NZ))
    N_history[0], P_history[0] = N, P

    for frame in range(1, n_frames):
        for s in range(steps_per_frame):
            N, P = step(N, P, ((frame - 1) * steps_per_frame + s) * dt)
        N_history[frame], P_history[frame] = N, P

    days = np.arange(n_frames) * steps_per_frame * dt / 86400
    res = Result(days, depth, N_history, P_history)
    if save_as: np.savez(save_as, **res._asdict())
    return res


def load(path):
    d = np.load(path)
    return Result(*[d[k] for k in Result._fields])


if __name__ == "__main__":      # python sim.py [params_file] [save_name]
    run(*sys.argv[1:3])