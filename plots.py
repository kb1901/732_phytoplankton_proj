import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Every function takes a Result (days, depth, N, P), returns (fig, ax)

def profiles(res, days):
    fig, ax = plt.subplots(1, 2, sharey=True)
    for d in days:
        k = np.argmin(abs(res.days - d))
        ax[0].plot(res.N[k], res.depth, label=f"day {res.days[k]:.0f}")
        ax[1].plot(res.P[k], res.depth)
    ax[0].set_title("N"); ax[1].set_title("P"); ax[0].set_ylabel("Depth (m)"); ax[0].legend()
    return fig, ax

def hovmoller(res, var="P", zmin=None):
    fig, ax = plt.subplots()
    m = ax.pcolormesh(res.days, res.depth, getattr(res, var).T, shading="auto")
    fig.colorbar(m, label=var)
    if zmin is not None: ax.set_ylim(zmin, 0)
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Depth (m)")
    return fig, ax

def timeseries(res, depths=(-5, -50, -100), var="P"):
    fig, ax = plt.subplots()
    for d in depths:
        k = np.argmin(abs(res.depth - d))
        ax.plot(res.days, getattr(res, var)[:, k], label=f"{res.depth[k]:.0f} m")
    ax.set_xlabel("Time (days)"); ax.set_ylabel(var); ax.legend()
    return fig, ax

def integrated(res):
    """Depth integrals of N, P and N+P."""
    N, P = _trapz(res.N, res.depth, axis=1), _trapz(res.P, res.depth, axis=1)
    fig, ax = plt.subplots()
    ax.plot(res.days, N, label="N"); ax.plot(res.days, P, label="P"); ax.plot(res.days, N + P, "k--", label="N+P")
    ax.set_xlabel("Time (days)"); ax.legend()
    return fig, ax

def animate(res, interval=100):
    fig, ax = plt.subplots()
    def draw(i):
        ax.clear(); ax.plot(res.N[i], res.depth, label="N"); ax.plot(res.P[i], res.depth, label="P")
        ax.set_title(f"{res.days[i]:.1f} days"); ax.legend()
    return animation.FuncAnimation(fig, draw, frames=len(res.days), interval=interval)   # keep a reference