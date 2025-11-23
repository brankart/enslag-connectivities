#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runseq_swiose.py
Runs particle advection for an ensemble of CROCO netCDF files.

"""

# ================================================================
# ENVIRONMENT
# ================================================================

import os
import errno
from datetime import timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import parcels
import tools as tl


# ================================================================
# GLOBAL PARAMETERS
# ================================================================

run_mode = "batchjob" # set to "frontend" if no job is needed

force_reprocess = False # set to True to overwrite existing outputs
ens_id = "ini" # "ini" or "GLS" or "CD"
file_type = "avg"
year = "2017"

nens = 30
ensemble = [1] #range(1,nens+1)
npart = 1
dt = 3600  # s = 1 day
dpt = -10
zone = "Glorioso"
radius = 0  # km, 0 to generate 1 particle on lat_ref, lon_ref


# ================================================================
# PARTICLE SETTINGS
# ================================================================

if zone == "Glorioso":
    lon_ref, lat_ref = 46, -11
elif zone == "Tromelin":
    lon_ref, lat_ref = 54.5, -16
elif zone == "Mayotte":
    lon_ref, lat_ref = 44, -13.5 # 45, -13
elif zone == "Madagascar":
    lon_ref, lat_ref = 50, -21
elif zone == "Southern":
    lon_ref, lat_ref = 50, -30

particles = tl.generate_particles(lat_ref, lon_ref, radius, npart)
X, Y, Z = particles[:, 1], particles[:, 0], np.full(npart, dpt)


# ================================================================
# DATA SHAPING
# ================================================================

variables = {"U": "u", "V": "v", "W": "w", "H": "h", "Zeta": "zeta", "Cs_w": "Cs_w"}

lon_rho = "lon_rho"
lat_rho = "lat_rho"

dimensions = {"U": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "V": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "W": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "H": {"lon": lon_rho, "lat": lat_rho},
              "Zeta": {"lon": lon_rho, "lat": lat_rho, "time": "time"},
              "Cs_w": {"depth": "s_w"},}


# ================================================================
# ACCESS PATHS
# ================================================================

if run_mode == "batchjob":
    swio_dataset_folder = f'/lus/scratch/CT1/c1601279/tbellayer/SWIOSE_dev/run_swio2_stoens30_{year}_{ens_id}'
elif run_mode == "frontend":
    swio_dataset_folder = f'/lus/store/CT1/c1601279/lweiss/run_croco/SWIOSE_dev/run_swio2_stoens30_{year}_{ens_id}'

dir_bin = f"output/{ens_id}/{zone}/z{abs(dpt)}/{npart}part"
dir_fig = f"figures/{ens_id}/{zone}/z{abs(dpt)}/{npart}part"
file_ensmean_bin = f"traj_{npart}part_ENSMEAN_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.zarr"
file_ensmean_fig = f"traj_{npart}part_ENSMEAN_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.png"
path_ensmean_bin = os.path.join(dir_bin,file_ensmean_bin)
path_ensmean_fig = os.path.join(dir_fig,file_ensmean_fig)

try:
    os.makedirs(dir_fig)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for idx in ensemble :

    file_bin = f"traj_{npart}part_memb{idx:03d}_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.zarr"
    file_fig = f"traj_{npart}part_memb{idx:03d}_{zone}_z{abs(dpt)}_{year}_{ens_id}_dt{dt}.png"
    path_bin = os.path.join(dir_bin,file_bin)
    path_fig = os.path.join(dir_fig,file_fig)

    fname = f"{idx:03d}swiose_{file_type}.nc"
    file = os.path.join(swio_dataset_folder, fname)


# ================================================================
# SEQUENTIAL RUN
# ================================================================

    # to call the run_sequential function :
    # if __name__ == "__main__":
    #   run_sequential(force_reprocess=False, members=[1, 5, 10])

    # Skip logic
    if not force_reprocess and os.path.exists(path_bin):
        print(f"⏩ Skipping {fname}, already processed.")
        continue

    print(f"Processing {fname} ...")

    with xr.open_dataset(file) as ds:
        hc_val = ds.hc.values
        time = ds.time.values
        timespan = len(time)

    #timespan = 365#days
    print(f"Time span : {timespan} outputs, from {min(time)} to {max(time)}")
    fieldset = parcels.FieldSet.from_croco(file, variables, dimensions, hc=hc_val, allow_time_extrapolation=True)

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=X,
        lat=Y,
        depth=Z,
    )

    outputfile = pset.ParticleFile(name=path_bin, outputdt=timedelta(days=1))

    pset.execute(
        [parcels.AdvectionRK4_3D, tl.DeleteParticle],
        runtime=86400*timespan,
        dt=dt, # SWIOSE timestep (s)
        output_file=outputfile,
    )

    del fieldset, pset, outputfile

print("✅ Ensemble processing completed.")



