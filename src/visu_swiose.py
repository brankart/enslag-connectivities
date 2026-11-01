#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visu_swiose.py
Visualizes outputs from runseq_swiose.py and saves it to .png image.

"""

# ========================================================
# ENVIRONMENT 
# ========================================================

import os
import errno
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.collections as clc
import matplotlib.colors as clr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as mg
import parcels
import tools as tl
from config import (
    ENSEMBLE_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    EXPT,
    N_MEMBERS,
)


# ========================================================
# GLOBAL SETTINGS
# ========================================================

force_resave = True # set to True to save figure(s)
map_bathy = False
plot_spaghetti = True
plot_colortime = False
map_colortime = False
map_colortime_bathy = False
plot_colortraj = False
map_colortraj = False
map_colortraj_bathy = False
plot_dt_comparison = False
plot_particles = (plot_spaghetti or plot_colortime or map_colortime or map_colortime_bathy
                  or plot_colortraj or map_colortraj or map_colortraj_bathy)
nens = 3
ensemble = range(1,nens+1)
#sample = [1,2,3,4,5,6] # members from which to plot
#sample = [1,2,3] # members from which to plot
sample = [2] # members from which to plot
npart = 1
dt = 300
dtL = [300,600,1200,1800,3600]
dpt = -10

# ========================================================
# ACCESS PATHS
# ========================================================

file_bathy_bin = str(OUTPUT_DIR / EXPT / "bathy.zarr")
file_bathy_fig = str(FIGURES_DIR / EXPT / "bathy.png")

dir_bin = str(OUTPUT_DIR / EXPT)
dir_fig = str(FIGURES_DIR / EXPT)

file_dt_bin = [f"traj_member_001.zarr/proc00.zarr" for idx in dtL]
file_dt_fig = f"traj_member_001.png"

file_spag_fig = f"traj_FULLENS_spaghetti.png"

file_colortime_fig = f"traj_FULLENS_colortime.png"
file_colortime_map = f"traj_FULLENS_colortime_map.png"
file_colortime_map_bathy = f"traj_FULLENS_colortime_map_bathy.png"
file_colortime_map_bathy_sample = f"traj_FULLENS_colortime_map_bathy_6memb.png"

file_colortraj_fig = f"traj_FULLENS_colortraj.png"
file_colortraj_map = f"traj_FULLENS_colortraj_map.png"
file_colortraj_map_bathy = f"traj_FULLENS_colortraj_map_bathy.png"

path_dt_bin = [os.path.join(dir_bin,t) for t in file_dt_bin]
path_dt_fig = os.path.join(dir_fig,file_dt_fig)

path_spag_fig = os.path.join(dir_fig,file_spag_fig)

path_colortime_fig = os.path.join(dir_fig,file_colortime_fig)
path_colortime_map = os.path.join(dir_fig,file_colortime_map)
path_colortime_map_bathy = os.path.join(dir_fig,file_colortime_map_bathy)
path_colortime_map_bathy_sample = os.path.join(dir_fig,file_colortime_map_bathy_sample)

path_colortraj_fig = os.path.join(dir_fig,file_colortraj_fig) 
path_colortraj_map = os.path.join(dir_fig,file_colortraj_map)
path_colortraj_map_bathy = os.path.join(dir_fig,file_colortraj_map_bathy)

file_bin = [f"traj_member_{idx:03d}.zarr/proc{idx-1:02d}.zarr" for idx in sample]
file_fig = [f"traj_member_{idx:03d}.png" for idx in sample]
path_bin = [os.path.join(dir_bin,f) for f in file_bin]
path_fig = [os.path.join(dir_fig,f) for f in file_fig]

def main():
    try:
        os.makedirs(dir_fig, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # ========================================================
    # LOADING DATA (OPTIONAL : LAND MASK)
    # ========================================================

    lon_reg, lat_reg, h_reg = None, None, None
    if map_bathy or map_colortime_bathy:
        if os.path.exists(file_bathy_bin):
            with xr.open_zarr(file_bathy_bin) as bathy_swiose:
                mask = bathy_swiose['mask_rho']  # .values
                lon = bathy_swiose['lon'].values
                lat = bathy_swiose['lat'].values
                h = bathy_swiose['h'].values
                # Checking for aberrant values in CROCO mappables and re-gridding
                lon_reg, lat_reg, mask_reg = tl.regrid_curv2reg(lon, lat, mask)
                lon_reg, lat_reg, h_reg = tl.regrid_curv2reg(lon, lat, h)
        else:
            print(f" *** Warning: {file_bathy_bin} not found. ***")

    if plot_particles:
        data_xarray = [xr.open_zarr(path_bin[idx]) for idx in range(len(sample))]
        x = [data_xarray[idx]["lon"].values for idx in range(len(data_xarray))]
        y = [data_xarray[idx]["lat"].values for idx in range(len(data_xarray))]
        z = [data_xarray[idx]["z"].values for idx in range(len(data_xarray))]  # depth of particles
        x0 = x[0][:, 0]  # first indices of each trajectory
        y0 = y[0][:, 0]

    # ========================================================
    # FIGURE PRODUCTION
    # ========================================================

    if map_bathy and force_resave:
        tl.map_bathy(lon_reg, lat_reg, h_reg, savepath=file_bathy_fig)
    elif map_bathy:
        tl.map_bathy(lon_reg, lat_reg, h_reg)

    if plot_spaghetti and force_resave:
        tl.spaghetti(x, y, x0, y0, npart, ensemble, savepath=path_spag_fig)
    elif plot_spaghetti:
        tl.spaghetti(x, y, x0, y0, npart, ensemble)


    if plot_colortime and force_resave:
        tl.colorplot_time(x, y, x0, y0, npart, ensemble, savepath=path_colortime_fig)
    elif plot_colortime:
        tl.colorplot_time(x, y, x0, y0, npart, ensemble)

    if plot_colortraj and force_resave:
        tl.colorplot_traj(x, y, x0, y0, npart, ensemble, savepath=path_colortraj_fig)
    elif plot_colortraj:
        tl.colorplot_traj(x, y, x0, y0, npart, ensemble)


    if map_colortime and force_resave:
        tl.colormap_time(x, y, x0, y0, npart, ensemble, savepath=path_colortime_map)
    elif map_colortime:
        tl.colormap_time(x, y, x0, y0, npart, ensemble)

    if map_colortraj and force_resave:
        tl.colormap_traj(x, y, x0, y0, npart, ensemble, savepath=path_colortraj_map)
    elif map_colortraj:
        tl.colormap_traj(x, y, x0, y0, npart, ensemble)


    if plot_dt_comparison:
        # Visualize trajectories along several timesteps
        data_dt = [xr.open_zarr(path_dt_bin[idx]) for idx in range(len(dtL))]
        x_dt = [data_dt[idx]["lon"].values for idx in range(len(data_dt))]
        y_dt = [data_dt[idx]["lat"].values for idx in range(len(data_dt))]
        z_dt = [data_dt[idx]["z"].values for idx in range(len(data_dt))]  # depth of particles
        x0_dt = x_dt[0][:, 0]  # first indices of each trajectory
        y0_dt = y_dt[0][:, 0]

        tl.colormap_traj_dt(x_dt, y_dt, x0_dt, y0_dt, npart, dtL)


    if map_colortime_bathy and force_resave and sample == ensemble:
        tl.colormap_time_bathy(x, y, x0, y0, npart, ensemble, dpt, lon_reg=lon_reg, lat_reg=lat_reg,
                               h_reg=h_reg, savepath=path_colortime_map_bathy)
    elif map_colortime_bathy and force_resave and sample != ensemble:
        tl.colormap_time_bathy(x, y, x0, y0, npart, sample, dpt,
                               lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg, savepath=path_colortime_map_bathy_sample)
    elif map_colortime_bathy and sample == ensemble:
        tl.colormap_time_bathy(x, y, x0, y0, npart, ensemble, dpt,
                               lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg)
    elif map_colortime_bathy and sample != ensemble:
        tl.colormap_time_bathy(x, y, x0, y0, npart, sample, dpt,
                               lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg)

    print()
    print(" *** Done ! *** ")


if __name__ == "__main__":
    main()



