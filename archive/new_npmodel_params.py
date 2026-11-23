# --- Physics to include ---
GROWTH_ENABLED = True     # nutrient uptake / phytoplankton growth & mortality
SINKING_ENABLED = False   # phytoplankton sinking (advection)
MIXING_ENABLED = True     # vertical turbulent mixing (diffusion)
SHADING_ENABLED = False   # whether to add shading due to phytoplankton

# --- Grid & timestepping ---
DEPTH = 250.0           # water column depth (m)
Nz = 64                 # number of depth levels
N_FRAMES = 500          # number of saved snapshots
STEPS_PER_FRAME = 100   # integration substeps between snapshots
dt = 900                # substep length (s)

# --- BCs ---
N_BOTTOM = 1.0
P_BOTTOM = 0.0
Z_BOTTOM = 0.0
N_BOTTOM_NEUMANN = False

# --- Model parameters ---
GROWTH_RATE = 20 / 86400       # max growth rate (1/s)
MORTALITY_RATE = 0.1 / 86400   # mortality / remineralization rate of phytoplankton (1/s)
GRAZING_RATE = 9.26e-6 # max grazing rate (1/s)
ZOO_EXCRETION_RATE = 5.80e-7 # zooplankton excretion rate (1/s)
ZOO_MORTALITY_RATE = 2.31e-6 # zooplankton mortality rate (1/s) (m^3 / (mmol * s))
AZ = 0.7 # assimilated food fraction by zooplankton
FE = 0.2 # export ratio
LIGHT_ATTENUATION = 0.05       # how fast light decays with depth (1/m)
KP = 0.05 # light attenuation due to phytoplankton stacking (self-shading) (m^3/(m mmol))

MIXING_RATE = 1e-4             # vertical turbulent diffusivity (m^2/s)
SINK_SPEED = -0.5 / 86400      # phytoplankton sinking speed (m/s, negative = downward)

# --- Seasonal surface nutrient supply: oscillates between 0 and 2*FLUX_AMPLITUDE ---
FLUX_AMPLITUDE = 0 #1e-5
FLUX_PERIOD = 365 * 86400      # one year, in seconds

# ---- Save or not ----
SAVE = True