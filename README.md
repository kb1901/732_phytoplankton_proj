# 1D Vertical Biogeochemical NP Model

A one-dimensional vertical biogeochemical model for the Nutrient–Phytoplankton (NP) cycle.
Simulates the vertical distribution of a water column and time evolution of N and P under various
physical processes, including: 
- light-limited growth
- mortality
- sinking
- self-shading
- time-dependent envirnomental forcings

## Project Structure
- `sim.py` - Core numerical solver using operator splitting (Lie or Strang). Implements RK4 for biology, Crank-Nicolson
for diffusion, and Heun's method and central difference for sinking. 
- `params.py` - Default physical, biological, grid, BC, and IC parameters.
- `forcing.py` - Time-series forcing functions 
- `plots.py` - Visualization helper functions
- `*.ipynb` - Exploratory notebooks 

## User Details
Install the necessary dependencies: 
```
pip install -r requirements.txt
```
To run the simulation:
```
 python sim.py [params_file] [save_name]
```
The simulation returns: 
```# days: (frames,), depth: (NZ,) in m (negative down), N and P: (frames, NZ)```

To override the default parameters, specify them as a function argument in the same way as `params.py`: 
```
sim.run( {PARAM_NAME}={PARAM_VALUE} )
```

By default, the params_file being read is `params.py` in this repo. You can change parameters in this file as desired

## Examples

Please see `notebook.ipynb` for example runs as well as plots!


