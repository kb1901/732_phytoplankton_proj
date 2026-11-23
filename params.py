from forcings import *

# --- Processes on/off ---
GROWTH_ENABLED = True     # nutrient uptake / phytoplankton growth & mortality
SINKING_ENABLED = False   # phytoplankton sinking (advection)
MIXING_ENABLED = True     # vertical turbulent mixing (diffusion)
SHADING_ENABLED = False   # self-shading by phytoplankton

# --- Solution order (operator-splitting) ---
# List of (process, fraction of the time step). Splitting run in this order; each process's fractions must sum to 1.
# A palindromic list (same forwards and backwards) is (second-order) Strang splitting.
SPLITTING = [("growth", 0.5), ("sinking", 0.5), ("mixing", 1.0), ("sinking", 0.5), ("growth", 0.5)]     # Strang
# SPLITTING = [("growth", 1), ("sinking", 1), ("mixing", 1)]                                            #Lie (first-order)
# SPLITTING = [("mixing", 0.5), ("sinking", 0.5), ("growth", 1.0), ("sinking", 0.5), ("mixing", 0.5)]   #Strang

# --- Grid & timestepping ---
DEPTH = 250.0           # water column depth (m)
NZ = 64                 # number of depth levels
DT_SECONDS = 900        # time step (s)
T_END_DAYS = 520.0      # total simulation time (days)
SAVE_EVERY_DAYS = 1.0   # time between saved snapshots (days)

# --- Initial condition ---
N_INIT = 0.5
P_INIT = 0.5

# --- Boundary conditions ---
N_BOT_BC = "neumann"             # "neumann" or "dirichlet"
N_BOT_VALUE = constant(1.0)      # Dirichlet value of N (number or function of t); ignored if "neumann"
P_BOT_VALUE = 0.0                # P at the bottom (fixed)

# --- Model parameters ---
GROWTH_RATE = 20 / 86400       # G: max growth rate (1/s)
MORTALITY_RATE = 0.1 / 86400   # D: mortality / remineralization rate (1/s)
LIGHT_ATTENUATION = 0.05       # K: how fast light decays with depth (1/m)
KAPPA = 1e-4                   # kappa: vertical turbulent diffusivity (m^2/s)
SINK_SPEED = -0.5 / 86400      # W: phytoplankton sinking speed (m/s). Negative = downward
SHADING_COEFF = 0.03           # self-shading by phytoplankton (1/m per unit P); only used if SHADING_ENABLED

# --- Forcings ---
# functions of time t in seconds (build them with forcings.py; combine multiple forcings with add(...))
SURFACE_FLUX = constant(0.0)        # TOP: N flux into the surface
LIGHT_FORCING = constant(1.0)       # GROWTH: multiplier on surface light;
MORTALITY_FORCING = constant(1.0)   # GROWTH: multiplier on MORTALITY_RATE
