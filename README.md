# 1D Vertical Biogeochemical NP Model

A one-dimensional vertical biogeochemical model for the Nutrient–Phytoplankton (NP) cycle, based largely on the work by Hodges
and Rudnik (2004). Simulates the vertical distribution of a water column and time evolution of N and P under various
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

## Getting Started
Install the necessary dependencies: 
```
pip install -r requirements.txt
```
Run the simulation:
```
python sim.py
```




