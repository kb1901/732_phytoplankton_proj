# --- Physics to include ---
GROWTH_ENABLED = True    # nutrient uptake / phytoplankton growth & mortality
SINKING_ENABLED = False   # phytoplankton sinking (advection)
MIXING_ENABLED = True    # vertical turbulent mixing (diffusion)
SHADING_ENABLED = False   # whether to add shading due to phytoplankton

# --- Grid & timestepping ---
DEPTH = 250.0           # water column depth (m)
Nz = 64                 # number of depth levels
N_FRAMES = 500          # number of saved snapshots
STEPS_PER_FRAME = 100   # integration substeps between snapshots
dt = 900               # substep length (s)

# --- BCs ---
N_BOTTOM = 1.0
P_BOTTOM = 0.0
N_BOTTOM_NEUMANN = True

# --- Model parameters ---
GROWTH_RATE = 20 / 86400       # max growth rate (1/s)
MORTALITY_RATE = 0.1 / 86400   # mortality / remineralization rate (1/s)
LIGHT_ATTENUATION = 0.05       # how fast light decays with depth (1/m)
KP = 0.03 # light attenuation due to phytoplankton stacking (self-shading) (m^3/(m mmol))

MIXING_RATE = 1e-4             # vertical turbulent diffusivity (m^2/s)
SINK_SPEED = -0.5 / 86400      # phytoplankton sinking speed (m/s, negative = downward)

# --- Seasonal surface nutrient supply: oscillates between 0 and 2*FLUX_AMPLITUDE ---
FLUX_AMPLITUDE = 0 #1e-5
FLUX_PERIOD = 365 * 86400      # one year, in seconds