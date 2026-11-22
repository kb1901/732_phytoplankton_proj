import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
Nz = 32
numouts = 500
numsteps = 1000
dt = 1000

# parameters
Gdays = 20
Ddays = .1
G = Gdays * 1 / (24 * 60 * 60)
D = Ddays * 1 / (24 * 60 * 60)
kappa = 1e-4
K = 0.05
Wdays = -.5
W = Wdays * 1 / (24 * 60 * 60)

def mu(N, P, z):
    return G*N*P*np.exp(K*z) - D*P

# storage
Ns = np.zeros((numouts, Nz))
Ps = np.zeros((numouts, Nz))
zs = np.linspace(-250, 0, Nz)
dz = zs[1]-zs[0]

# ICs:
Ns[0,:] = np.ones(Nz)*.5
Ps[0,:] = np.ones(Nz)*.5

# BCs:
Ns[0,0] = 1 # bottom
Ps[0,0] = 0 # bottom

N_curr = Ns[0,:].copy()
N_prev = Ns[0,:].copy()
P_curr = Ps[0,:].copy()
P_prev = Ps[0,:].copy()

sinking = True

for out in range(1,numouts):
    # iterate through time
    for n in range(numsteps):
        # BCs at bottom (Dirichlet for P):
        #N_curr[0] = 1
        P_curr[0] = 0
        # BCs at bottom (Neumann for N):
        N_curr[0] = (
            N_prev[0] - dt*mu(N_prev[0],P_prev[0],zs[0]) + 
            2*kappa*dt*(N_prev[1]-N_prev[0])/(pow(dz,2))
        )
        if (not sinking):
            # iterate through z
            for i in range(1,Nz-1):
                N_curr[i] = (
                    N_prev[i] - dt*mu(N_prev[i],P_prev[i],zs[i]) + 
                    kappa*dt*(N_prev[i+1]-2*N_prev[i] + N_prev[i-1])/(pow(dz,2))
                )
                P_curr[i] = (
                    P_prev[i] + dt*mu(N_prev[i],P_prev[i],zs[i]) + 
                    kappa*dt*(P_prev[i+1]-2*P_prev[i] + P_prev[i-1])/(pow(dz,2))
                )
            # BCs at top ("Neumann")
            N_curr[Nz-1] = (
                N_prev[Nz-1] - dt*mu(N_prev[Nz-1],P_prev[Nz-1],zs[Nz-1]) + 
                2*kappa*dt*(N_prev[Nz-2]-N_prev[Nz-1])/(pow(dz,2))
            )
            P_curr[Nz-1] = (
                P_prev[Nz-1] + dt*mu(N_prev[Nz-1],P_prev[Nz-1],zs[Nz-1]) + 
                2*kappa*dt*(P_prev[Nz-2]-P_prev[Nz-1])/(pow(dz,2))
            )
        elif (sinking):
            # iterate through z
            for i in range(1,Nz-1):
                N_curr[i] = (
                    N_prev[i] - dt*mu(N_prev[i],P_prev[i],zs[i]) + 
                    kappa*dt*(N_prev[i+1]-2*N_prev[i] + N_prev[i-1])/(pow(dz,2))
                )
                P_curr[i] = (
                    P_prev[i] + dt*mu(N_prev[i],P_prev[i],zs[i]) + 
                    kappa*dt*(P_prev[i+1]-2*P_prev[i] + P_prev[i-1])/(pow(dz,2)) -
                    dt*W*(P_prev[i+1]-P_prev[i-1])/(2*dz)
                )
            # BCs at top ("Neumann")
            N_curr[Nz-1] = (
                N_prev[Nz-1] - dt*mu(N_prev[Nz-1],P_prev[Nz-1],zs[Nz-1]) + 
                2*kappa*dt*(N_prev[Nz-2]-N_prev[Nz-1])/(pow(dz,2))
            )
            P_curr[Nz-1] = (
                P_prev[Nz-1] + dt*mu(N_prev[Nz-1],P_prev[Nz-1],zs[Nz-1]) + 
                2*kappa*dt*(P_prev[Nz-2] + (W*P_prev[Nz-1]*dz/kappa) - P_prev[Nz-1])/(pow(dz,2)) -
                dt*W*(W*P_prev[Nz-1]/kappa)
            )
        N_prev = N_curr.copy()
        P_prev = P_curr.copy()
    Ns[out,:] = N_curr.copy()
    Ps[out,:] = P_curr.copy()

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.plot(Ns[frame,:],zs)
    ax.plot(Ps[frame,:],zs)
    ax.set_title(f"{round(frame*dt)}")

ani = animation.FuncAnimation(fig=fig, func=update, frames=numouts, interval=100)
plt.show()
ani.save("movi.gif", writer="pillow")