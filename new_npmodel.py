import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
Nz = 32
numouts = 500
numsteps = 1000
dt = 2000

# parameters
Gdays = 20
Ddays = 0.1
G = Gdays / (24 * 60 * 60)
D = Ddays / (24 * 60 * 60)
kappa = 1e-5
K = 0.05
Wdays = -0.5
W = Wdays / (24 * 60 * 60)

# Forcing
F_amp = 1e-6
omega = 2 * np.pi / (365 * 24 * 60 * 60)


def N_flux(t):
    return F_amp * (1 + np.sin(omega * t))


zs = np.linspace(-250, 0, Nz)
dz = zs[1] - zs[0]
alpha = kappa * dt / (dz ** 2)
beta = dt * W / (2 * dz)
exp_Kz = np.exp(K * zs)


def mu_vec(N, P, exp_Kz_arr):
    return G * N * P * exp_Kz_arr - D * P


# storage
Ns = np.zeros((numouts, Nz))
Ps = np.zeros((numouts, Nz))

# ICs
Ns[0, :] = 0.1
Ps[0, :] = 0.1

# BCs
Ns[0, 0] = 1  # bottom
Ps[0, 0] = 0  # bottom

N_curr = Ns[0, :].copy()
N_prev = Ns[0, :].copy()
P_curr = Ps[0, :].copy()
P_prev = Ps[0, :].copy()

sinking = True

W_dz_kappa = W * dz / kappa
W_sq_kappa = dt * (W ** 2) / kappa

for out in range(1, numouts):
    for n in range(numsteps):
        t = ((out - 1) * numsteps + n) * dt

        mu_val = mu_vec(N_prev, P_prev, exp_Kz)

        # Bottom BCs (Dirichlet)
        N_curr[0] = 1.0
        P_curr[0] = 0.0

        N_curr[1:-1] = (N_prev[1:-1] - dt * mu_val[1:-1] +
                        alpha * (N_prev[2:] - 2 * N_prev[1:-1] + N_prev[:-2]))
        P_curr[1:-1] = (P_prev[1:-1] + dt * mu_val[1:-1] +
                        alpha * (P_prev[2:] - 2 * P_prev[1:-1] + P_prev[:-2]))

        N_curr[-1] = (N_prev[-1] - dt * mu_val[-1] +
                      2 * alpha * (N_prev[-2] - N_prev[-1]) +
                      2 * dt * N_flux(t) / dz)
        P_curr[-1] = (P_prev[-1] + dt * mu_val[-1] +
                      2 * alpha * (P_prev[-2] - P_prev[-1]))

        if sinking:
            P_curr[1:-1] -= beta * (P_prev[2:] - P_prev[:-2])
            P_curr[-1] += 2 * alpha * (P_prev[-1] * W_dz_kappa) - (P_prev[-1] * W_sq_kappa)

        N_prev, N_curr = N_curr, N_prev
        P_prev, P_curr = P_curr, P_prev

    Ns[out, :] = N_prev
    Ps[out, :] = P_prev

fig, ax = plt.subplots()


def update(frame):
    ax.clear()
    ax.plot(Ns[frame, :], zs, label="Nutrients (N)")
    ax.plot(Ps[frame, :], zs, label="Phytoplankton (P)")
    ax.set_title(f"Time: {round(frame * numsteps * dt / (24 * 3600), 1)} Days")
    ax.legend()


ani = animation.FuncAnimation(fig=fig, func=update, frames=numouts, interval=100)
plt.show()
ani.save("movi.gif", writer="pillow")