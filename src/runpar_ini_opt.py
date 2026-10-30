#!/usr/bin/env python3
# ================================================================
# IMPORTS
# ================================================================

import os
import numpy as np
import xarray as xr
from datetime import timedelta
from mpi4py import MPI
import parcels
# import tools as tl


# ================================================================
# MPI SETUP
# ================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ================================================================
# USER PARAMETERS
# ================================================================

from config import (
    EXPT,
    N_MEMBERS,
    PARTICLES_PER_POINT,
    GAUSSIAN_STD_DEG,
    DT,
    ENSEMBLE_DIR,
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    SPOTS_FILE,
)

OUTPUT_DT = timedelta(days=1)
OUTPUT_DIR = str(BASE_OUTPUT_DIR / EXPT)
FILE_PATTERN = "{:03d}swiose_avg.nc"


# ================================================================
# CROCO DATA SHAPING
# ================================================================

variables = {"U": "u", "V": "v", "W": "w", "H": "h",
             "Zeta": "zeta", "Cs_w": "Cs_w"}

lon_rho = "lon_rho"
lat_rho = "lat_rho"

dimensions = {"U": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "V": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "W": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
              "H": {"lon": lon_rho, "lat": lat_rho},
              "Zeta": {"lon": lon_rho, "lat": lat_rho, "time": "time"},
              "Cs_w": {"depth": "s_w"}}


# ================================================================
# KERNELS
# ================================================================
 
def DeleteParticle(particle, fieldset, time):
    if particle.state >= 50:
        particle.delete()
 
 
# ================================================================
# LOAD RELEASE POINTS
# ================================================================
 
def load_release_points():
    data = np.loadtxt(SPOTS_FILE, comments='#', ndmin=2)
    lon = data[:, 0]
    lat = data[:, 1]
    if data.shape[1] >= 3:
        depth = data[:, 2]
    else:
        depth = np.zeros_like(lon)
 
    return lon, lat, depth
 
 
# ================================================================
# GENERATE PARTICLE CLOUD
# ================================================================
 
def create_particle_cloud(lon, lat, depth):
    """
    Generate a particle cloud around each release point.
    If PARTICLES_PER_POINT > 1: 2D Gaussian distribution (lon, lat)
    centred on the release point, with standard deviation GAUSSIAN_STD_DEG in degrees.
    If PARTICLES_PER_POINT == 1: a single particle placed exactly on the release point.
    """

    lon_list = []
    lat_list = []
    depth_list = []

    for lo, la, de in zip(lon, lat, depth):
        if PARTICLES_PER_POINT > 1:
            lo_cloud, la_cloud = np.random.multivariate_normal(
                mean=[lo, la],
                cov=[[GAUSSIAN_STD_DEG**2, 0],
                     [0,                   GAUSSIAN_STD_DEG**2]],
                size=PARTICLES_PER_POINT
            ).T
        else:
            lo_cloud = np.array([lo])
            la_cloud = np.array([la])

        lon_list.append(lo_cloud)
        lat_list.append(la_cloud)
        depth_list.append(np.full(PARTICLES_PER_POINT, de))

    lon_all = np.concatenate(lon_list)
    lat_all = np.concatenate(lat_list)
    depth_all = np.concatenate(depth_list)

    return lon_all, lat_all, depth_all
 
 
# ================================================================
# CREATE FIELDSET
# ================================================================
 
def create_fieldset(file):
 
    with xr.open_dataset(file) as ds:
        hc_val = ds.hc.values
        timespan = len(ds.time)
 
    fieldset = parcels.FieldSet.from_croco(
        file, variables, dimensions,
        hc=hc_val,
        allow_time_extrapolation=True,
        deferred_load=True
    )
 
    return fieldset, timespan
 
 
# ================================================================
# RUN ONE ENSEMBLE MEMBER
# ================================================================
 
def run_member(member, lon, lat, depth):
 
    fname = FILE_PATTERN.format(member)
    file = os.path.join(ENSEMBLE_DIR, fname)
 
    if not os.path.exists(file):
        print(f"[rank {rank}] Missing {file}", flush=True)
        return
 
    print(f"[rank {rank}] Running member {member}", flush=True)
 
    fieldset, timespan = create_fieldset(file)
 
    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=lon, lat=lat, depth=depth
    )
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    output_name = os.path.join(OUTPUT_DIR, f"traj_member_{member:03d}.zarr")
    output_file = pset.ParticleFile(
        name=output_name,
        outputdt=OUTPUT_DT,
        chunks=(10000, 1)
    )
 
    pset.execute(
        [parcels.AdvectionRK4_3D, DeleteParticle],
        runtime=86400 * timespan,
        dt=DT,
        output_file=output_file
    )
 
    del fieldset
    del pset
 
    print(f"[rank {rank}] Finished member {member}", flush=True)
 
 
# ================================================================
# MAIN PROGRAM
# ================================================================
 
def main():
 
    if rank == 0:
        print("\nLoading release points", flush=True)
 
    lon, lat, depth = load_release_points()
    lonp, latp, depthp = create_particle_cloud(lon, lat, depth)
 
    if rank == 0: # for debug
        print(f"Release points:            {len(lon)}")
        print(f"Particles per point:       {PARTICLES_PER_POINT}")
        print(f"Total particles per member:{len(lonp)}")
        print(f"MPI ranks:                 {size}")
        print(f"Members per rank:          {N_MEMBERS // size} "
              f"(+{N_MEMBERS % size} ranks carry 1 extra)", flush=True)
 
    members = list(range(1, N_MEMBERS + 1))
    local_members = members[rank::size]
 
    print(f"[rank {rank:02d}] Members assigned: {local_members}", flush=True)
 
    for member in local_members:
        run_member(member, lonp, latp, depthp)
 
    comm.Barrier()
 
    if rank == 0:
        print("\nAll simulations completed.", flush=True)
 
 
# ================================================================
# ENTRY POINT
# ================================================================
 
if __name__ == "__main__":
    main()
