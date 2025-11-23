# BRIDGES-AVATAR-lagrangian: An introduction to Lagrangian post-modelling and processing

This preliminary work aims at providing useful tests on lagrangian modelling tools that take ocean circulation runs as inputs, within the [BRIDGES-AVATAR](https://www.bridges-wio.com/fr/projet/pc-3-avatar/) research project workframe. The idea is to provide a "starter" set of tools to explore the outputs of chosen simulations, and get used to the associated Lagrangian post-pocessing methods.

<img src="figures_SWIOSE/ini/Glorioso/z1000/1part/traj_1part_FULLENS_Glorioso_z1000_2019_ini_dt300_colortime_map_bathy.png" alt="drawing" width="600"/>
**Fig.1: Example of particle drifting from the Glorioso archipelago, starting on Jan 1st 2019 (30 particles on a 30-member ensemble simulation, one particle on each member.)**



## CONTEXT, BACKGROUND AND OBJECTIVES

In the BRIDGES-AVATAR workframe, we take the SouthWest Indian Ocean (SWIO) as a testbed. This repository standing as a starting point, we made tests on the SWIOSE (SouthWest Indian Ocean Stochastic Ensembles), developed by Lisa Weiss and Jean-Michel Brankart in the [MEOM team, IGE, Grenoble](https://meom-group.github.io/). The goal is to assess the importance of several sources of uncertainty that affect the surface circulation variability in the SWIO region, in anticipation of the MARIO cruise to be conducted in 2028. 


## INSTALLATION INSTRUCTIONS


### OceanParcels

  There is a wide offer of lagrangian free, open-source software, but we decided to use the [OceanParcels](https://parcels-code.org/) simulator. The main reason is that it is distributed as a free python module, which makes it very easy to install, set up and customize for any personnal experiment. One inconveninent might be its slower computational time when compared to other simulators, due to the language architecture; However it is easy to perform customized, parallel computing on clusters with it. Yu can create your own parcels python environment by following [this tutorial.](https://docs.parcels-code.org/en/latest/installation.html).


### Other useful tools

+ **[Install VirtualFleet:](https://github.com/euroargodev/VirtualFleet)** a free, open-source set of python classes and routines, developed on top of OceanParcels, to mimick the behavior and deployment strategies of ARGO floats.

+ **[Install zarr2netcdf:](https://github.com/daanreijnders/zarr2netcdf)** a free, open-source python protocol to convert .zarr data stores (used by default in oceanparcels and its derivatives) to netcdf files. This can be very convenient for data compression, long-term storage and transfer.


### Quickstart with a few basic tools

### Cruise reproduction or planning



## SOURCES 

+ **[getvars_swiose.py:](https://github.com/TinoBELLAYER/BRIDGES-AVATAR-lagrangian/blob/main/getvars_swiose.py)** extract specific variables from the input simulations, i.e. ssh or bathymetry, to visualize them alongside OceanParcels outputs. 
+ **[runseq_swiose.py:](https://github.com/TinoBELLAYER/BRIDGES-AVATAR-lagrangian/blob/main/runseq_swiose.py)** compute lagrangian trajectories with ocean parcels, using SWIOSE CROCO outputs.
+ **[visu_swiose.py:](https://github.com/TinoBELLAYER/BRIDGES-AVATAR-lagrangian/blob/main/visu_swiose.py)** produce figures to visualize *runseq_swiose.py* outputs.
+ **[tools.py:](https://github.com/TinoBELLAYER/BRIDGES-AVATAR-lagrangian/blob/main/tools.py)** a set of functions to perform a wide range of tasks involved in this work, such as particle generation, and specific visualization.


