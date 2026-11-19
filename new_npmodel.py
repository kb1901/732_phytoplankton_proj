import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# --- Physics to include ---
GROWTH_ENABLED = True    # nutrient uptake / phytoplankton growth & mortality
SINKING_ENABLED = True   # phytoplankton sinking (advection)
MIXING_ENABLED = True    # vertical turbulent mixing (diffusion)

# --- Grid & timestepping ---
DEPTH = 250.0           # water column depth (m)
Nz = 64                 # number of depth levels
N_FRAMES = 500          # number of saved snapshots
STEPS_PER_FRAME = 100   # integration substeps between snapshots
dt = 900               # substep length (s)


GROWTH_RATE = 20 / 86400       # max growth rate (1/s)
MORTALITY_RATE = 0.1 / 86400   # mortality / remineralization rate (1/s)
LIGHT_ATTENUATION = 0.05       # how fast light decays with depth (1/m)

MIXING_RATE = 1e-4             # vertical turbulent diffusivity (m^2/s)
SINK_SPEED = -0.5 / 86400      # phytoplankton sinking speed (m/s, negative = downward)

# --- Seasonal surface nutrient supply: oscillates between 0 and 2*FLUX_AMPLITUDE ---
FLUX_AMPLITUDE = 1e-5
FLUX_PERIOD = 365 * 86400      # one year, in seconds

N_BOTTOM = 1.0
P_BOTTOM = 0.0

def apply_bottom_bc(N, P):
    N[0], P[0] = N_BOTTOM, P_BOTTOM
    return N, P


def surface_nutrient_flux(t):
    return FLUX_AMPLITUDE * (1 + np.sin(2 * np.pi * t / FLUX_PERIOD))


# --- Grid ---
depth_levels = np.linspace(-DEPTH, 0, Nz)   # z=0 is the surface, z=-DEPTH is the bottom
dz = depth_levels[1] - depth_levels[0]
light_profile = np.exp(LIGHT_ATTENUATION * depth_levels)  # relative light level at each depth

# --- Mixing (diffusion) matrix, built once and factorized once ---
# Implicit Crank-Nicolson in depth, with:
#   bottom (z=-DEPTH): Dirichlet  -> value is held fixed by the caller
#   top    (z=0):      Neumann    -> zero ghost-point derivative
gamma = MIXING_RATE * dt / (2 * dz ** 2)

main_diag = (1 + 2 * gamma) * np.ones(Nz)
off_diag = -gamma * np.ones(Nz - 1)
mixing_matrix = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="lil")
mixing_matrix[0, 0], mixing_matrix[0, 1] = 1.0, 0.0
mixing_matrix[-1, -2] = -2 * gamma
mixing_solver = splu(mixing_matrix.tocsc())


def apply_growth(N, P, dt):
    """One RK4 step of nutrient uptake / phytoplankton growth and mortality."""
    def rates(N, P):
        growth = GROWTH_RATE * N * P * light_profile - MORTALITY_RATE * P
        return -growth, growth

    dN1, dP1 = rates(N, P)
    dN2, dP2 = rates(N + dt / 2 * dN1, P + dt / 2 * dP1)
    dN3, dP3 = rates(N + dt / 2 * dN2, P + dt / 2 * dP2)
    dN4, dP4 = rates(N + dt * dN3, P + dt * dP3)

    N_next = N + dt / 6 * (dN1 + 2 * dN2 + 2 * dN3 + dN4)
    P_next = P + dt / 6 * (dP1 + 2 * dP2 + 2 * dP3 + dP4)
    return N_next, P_next


def apply_sinking(P, dt):
    """One upwind-advection step moving phytoplankton downward."""
    courant = SINK_SPEED * dt / dz   # <= 0, since sinking is downward
    P_next = P.copy()
    P_next[:-1] -= courant * (P[1:] - P[:-1])
    P_next[-1] += courant * P[-1]
    return P_next


def apply_mixing(profile, top_flux=0.0):
    """One implicit (Crank-Nicolson) vertical mixing step. """
    rhs = np.empty_like(profile)
    rhs[1:-1] = gamma * profile[:-2] + (1 - 2 * gamma) * profile[1:-1] + gamma * profile[2:]
    rhs[0] = profile[0]   # keep the bottom value as given
    rhs[-1] = 2 * gamma * profile[-2] + (1 - 2 * gamma) * profile[-1] + top_flux
    return mixing_solver.solve(rhs)


# --- Initial conditions ---
N = np.full(Nz, 0.5)
P = np.full(Nz, 0.5)
N[0], P[0] = 1.0, 0.0   # bottom boundary values

N_history = np.zeros((N_FRAMES, Nz))
P_history = np.zeros((N_FRAMES, Nz))
N_history[0], P_history[0] = N, P

# --- Main time integration loop ---
for frame in range(1, N_FRAMES):
    for step in range(STEPS_PER_FRAME):
        t = ((frame - 1) * STEPS_PER_FRAME + step) * dt

        if GROWTH_ENABLED:
            N, P = apply_growth(N, P, dt)
            N, P = apply_bottom_bc(N, P)

        if SINKING_ENABLED:
            P = apply_sinking(P, dt)
            N, P = apply_bottom_bc(N, P)

        if MIXING_ENABLED:
            top_flux = gamma * (2 * dz / MIXING_RATE) * (
                surface_nutrient_flux(t) + surface_nutrient_flux(t + dt)
            )
            N = apply_mixing(N, top_flux=top_flux)
            P = apply_mixing(P, top_flux=0.0)
            N, P = apply_bottom_bc(N, P)

    N_history[frame], P_history[frame] = N, P

# --- Plotting & animation ---
def draw_frame(frame):
    ax.clear()
    ax.plot(N_history[frame], depth_levels, label="Nutrients (N)")
    ax.plot(P_history[frame], depth_levels, label="Phytoplankton (P)")
    days = frame * STEPS_PER_FRAME * dt / 86400
    ax.set_title(f"Time: {days:.1f} Days")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

# ani.save("movie.gif", writer="pillow")  # Uncomment to save


def plot_time_series(depth_index=-1):
    """Plot N and P at a single depth across the whole run (default: surface)."""
    days = np.arange(N_FRAMES) * STEPS_PER_FRAME * dt / 86400

    fig, (ax_N, ax_P) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(f"Time series at depth = {depth_levels[depth_index]:.0f} m")

    ax_N.plot(days, N_history[:, depth_index])
    ax_N.set_ylabel("Nutrients (N)")
    ax_N.grid(True, linestyle="--", alpha=0.5)

    ax_P.plot(days, P_history[:, depth_index])
    ax_P.set_ylabel("Phytoplankton (P)")
    ax_P.set_xlabel("Time (days)")
    ax_P.grid(True, linestyle="--", alpha=0.5)

    plt.show()

def plot_integrated():
    days = np.arange(N_FRAMES) * STEPS_PER_FRAME * dt / 86400
    fig, (ax_N, ax_P) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Integrated Time Series")

    ax_N.plot(days, np.sum(N_history, axis=1))
    ax_N.set_ylabel("Integrated Nutrients (N)")
    ax_N.grid(True, linestyle="--", alpha=0.5)

    ax_P.plot(days, np.sum(P_history, axis=1))
    ax_P.set_ylabel("Integrated Phytoplankton (P)")
    ax_P.set_xlabel("Time (days)")
    ax_P.grid(True, linestyle="--", alpha=0.5)

    plt.show()

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig=fig, func=draw_frame, frames=N_FRAMES, interval=100)
plt.show()
plot_integrated()
plot_time_series()  # surface by default; pass e.g. depth_index=0 for the bottom