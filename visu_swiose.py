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


# ========================================================
# GLOBAL SETTINGS
# ========================================================

force_resave = False # set to True to save figure(s)
map_bathy = False
map_ssh = False
map_ssh_multi = False
plot_spaghetti = False
plot_colortime = False
map_colortime = False
map_colortime_bathy = False
plot_colortraj = False
map_colortraj = False
map_colortraj_bathy = False
plot_particles = (plot_spaghetti or plot_colortime or map_colortime or map_colortime_bathy
                  or plot_colortraj or map_colortraj or map_colortraj_bathy)
nens = 30
ens_id = "ini" # "ini" or "GLS" or "CD"
year = "2019"
zone = "Glorioso"
ensemble = range(1,nens+1)
sample = [1,2,3,4,5,6] # members from which to plot ssh
npart = 1
dt = 300
dtL = [300,600,1200,1800,3600]
dpt = -10
sigma = 0.1
radius = 0  # km
if zone == "Glorioso":
    lon_ref, lat_ref = 46, -11
elif zone == "Tromelin":
    lon_ref, lat_ref = 54.5, -16
elif zone == "Mayotte":
    lon_ref, lat_ref = 45, -13
elif zone == "Madagascar":
    lon_ref, lat_ref = 50, -21
elif zone == "Southern":
    lon_ref, lat_ref = 50, -30


# ========================================================
# ACCESS PATHS
# ========================================================

swio_dataset_folder = f'/lus/store/CT1/c1601279/lweiss/run_croco/SWIOSE_dev/run_swio2_stoens30_2019_{ens_id}'
fname = [f"{idx:03d}swiose_avg.nc" for idx in sample]
file = [os.path.join(swio_dataset_folder, f) for f in fname]

file_bathy_bin = f"output/{ens_id}/bathy.zarr"
file_bathy_fig = f"figures/{ens_id}/bathy.png"
file_ssh_bin = [f"output/{ens_id}/modelvars/ssh/ssh_memb{idx:03d}_{year}_{ens_id}.zarr" for idx in sample]
file_ssh_fig = [f"figures/{ens_id}/modelvars/ssh/ssh_memb{idx:03d}_{year}_{ens_id}.png" for idx in sample]
file_ssh_multi_fig = f"figures/{ens_id}/modelvars/ssh/ssh_multi_{year}_{ens_id}.png"

dir_bin = f"output/{ens_id}/{zone}/z{abs(dpt)}/{npart}part"
dir_fig = f"figures/{ens_id}/{zone}/z{abs(dpt)}/{npart}part"

file_ensmean_bin = f"traj_{npart}part_ENSMEAN_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.zarr"
file_ensmean_fig = f"traj_{npart}part_ENSMEAN_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.png"

file_dt_bin = [f"traj_{npart}part_memb001_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{idx}.zarr" for idx in dtL]
file_dt_fig = f"traj_{npart}part_memb001_{zone}_z{abs(dpt)}_{year}_{ens_id}_dtmulti.png"

file_spag_fig = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_spaghetti.png"

file_colortime_fig = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortime.png"
file_colortime_map = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortime_map.png"
file_colortime_map_bathy = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortime_map_bathy.png"
file_colortime_map_bathy_sample = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortime_map_bathy_6memb.png"

file_colortraj_fig = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortraj.png"
file_colortraj_map = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortraj_map.png"
file_colortraj_map_bathy = f"traj_{npart}part_FULLENS_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}_colortraj_map_bathy.png"

path_ensmean_bin = os.path.join(dir_bin,file_ensmean_bin)
path_ensmean_fig = os.path.join(dir_fig,file_ensmean_fig)

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

file_bin = [f"traj_{npart}part_memb{idx:03d}_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.zarr" for idx in sample]
file_fig = [f"traj_{npart}part_memb{idx:03d}_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.png" for idx in sample]
path_bin = [os.path.join(dir_bin,f) for f in file_bin]
path_fig = [os.path.join(dir_fig,f) for f in file_fig]

try:
    os.makedirs(dir_fig)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(f"figures/{ens_id}/modelvars/ssh")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


# ========================================================
# LOADING DATA (OPTIONAL : LAND MASK)
# ========================================================

with xr.open_zarr(file_bathy_bin) as bathy_swiose :
    mask = bathy_swiose['mask_rho']#.values
    lon = bathy_swiose['lon'].values
    lat = bathy_swiose['lat'].values
    h = bathy_swiose['h'].values
    # h = bathy_swiose['h'].where(mask==1).values
    # Checking for aberrant values in CROCO mappables and re-gridding
    lon_reg, lat_reg, mask_reg = tl.regrid_curv2reg(lon,lat,mask)
    lon_reg, lat_reg, h_reg = tl.regrid_curv2reg(lon,lat,h)

if map_ssh or map_ssh_multi :
    ssh_ini_reg = []    
    for n in file_ssh_bin :
        ssh_n = xr.open_zarr(n)['zeta'].values[0,:,:]
        # ssh_ini = ssh_swiose['zeta'].where(mask==1).values[0,:,:]
        lon_reg, lat_reg, ssh_n_reg = tl.regrid_curv2reg(lon,lat,ssh_n)
        ssh_ini_reg.append(ssh_n_reg)

if plot_particles:
    data_xarray = [xr.open_zarr(path_bin[idx]) for idx in range(len(sample))]
    x = [data_xarray[idx]["lon"].values for idx in range(len(data_xarray))]
    y = [data_xarray[idx]["lat"].values for idx in range(len(data_xarray))]
    z = [data_xarray[idx]["z"].values for idx in range(len(data_xarray))] # depth of particles
    x0 = x[0][:,0] # first indices of each trajectory
    y0 = y[0][:,0]


# ========================================================
# FIGURE PRODUCTION
# ========================================================

if map_bathy and force_resave:
    tl.map_bathy(lon_reg, lat_reg, h_reg, savepath=file_bathy_fig)
elif map_bathy:
    tl.map_bathy(lon_reg, lat_reg, h_reg)

if map_ssh and force_resave:
    tl.map_ssh(lon_reg, lat_reg, ssh_ini_reg, savepath=file_ssh_fig)
elif map_ssh:
    tl.map_ssh(lon_reg, lat_reg, ssh_ini_reg)

if map_ssh_multi and force_resave:
    tl.map_ssh_multi(lon_reg, lat_reg, ssh_ini_reg[:6], savepath=file_ssh_multi_fig)
elif map_ssh_multi:
    tl.map_ssh_multi(lon_reg, lat_reg, ssh_ini_reg[:6])


if plot_spaghetti and force_resave :
    tl.spaghetti(x, y, x0, y0, npart, ensemble, savepath=path_spag_fig)
elif plot_spaghetti :
    tl.spaghetti(x, y, x0, y0, npart, ensemble)


if plot_colortime and force_resave :
    tl.colorplot_time(x, y, x0, y0, npart, ensemble, savepath=path_colortime_fig)
elif plot_colortime :
    tl.colorplot_time(x, y, x0, y0, npart, ensemble)

if plot_colortraj and force_resave :
    tl.colorplot_traj(x, y, x0, y0, npart, ensemble, savepath=path_colortraj_fig)
elif plot_colortraj :
    tl.colorplot_traj(x, y, x0, y0, npart, ensemble)


if map_colortime and force_resave :
    tl.colormap_time(x, y, x0, y0, npart, ensemble, savepath=path_colortime_map)
elif map_colortime:
    tl.colormap_time(x, y, x0, y0, npart, ensemble)

if map_colortraj and force_resave :
    tl.colormap_traj(x, y, x0, y0, npart, ensemble, savepath=path_colortraj_map)
elif map_colortraj:
    tl.colormap_traj(x, y, x0, y0, npart, ensemble)


# Visualize trajectories along several timesteps (function  to make)
data_dt = [xr.open_zarr(path_dt_bin[idx]) for idx in range(len(dtL))]
x = [data_dt[idx]["lon"].values for idx in range(len(data_dt))]
y = [data_dt[idx]["lat"].values for idx in range(len(data_dt))]
z = [data_dt[idx]["z"].values for idx in range(len(data_dt))] # depth of particles
x0 = x[0][:,0] # first indices of each trajectory
y0 = y[0][:,0]

tl.colormap_traj_dt(x, y, x0, y0, npart, dtL)



if map_colortime_bathy and force_resave and sample==ensemble :
    tl.colormap_time_bathy(x, y, x0, y0, npart, ensemble, dpt, lon_reg=lon_reg, lat_reg=lat_reg,
                           h_reg=h_reg, savepath=path_colortime_map_bathy)
elif map_colortime_bathy and force_resave and sample!=ensemble :
    tl.colormap_time_bathy(x, y, x0, y0, npart, sample, dpt,
                           lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg, savepath=path_colortime_map_bathy_sample)
elif map_colortime_bathy and sample==ensemble:
    tl.colormap_time_bathy(x, y, x0, y0, npart, ensemble, dpt,
                           lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg)
elif map_colortime_bathy and sample!=ensemble:
    tl.colormap_time_bathy(x, y, x0, y0, npart, sample, dpt,
                           lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg)

#if map_colortraj_bathy and force_resave :
#    tl.colormap_traj_bathy(x, y, x0, y0, npart, ensemble, dpt, background=False, lon_reg=lon_reg,
#                           lat_reg=lat_reg, h_reg=h_reg, savepath=path_colortraj_map_bathy)
#elif map_colortraj_bathy :
#    tl.colormap_traj_bathy(x, y, x0, y0, npart, ensemble, dpt, background=False,
#                           lon_reg=lon_reg, lat_reg=lat_reg, h_reg=h_reg)


print()
print(" *** Done ! *** ")



