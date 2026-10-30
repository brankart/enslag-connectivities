#!/usr/bin/env python3
# ================================================================
# IMPORTS
# ================================================================

import os
import sys
import numpy as np
import xarray as xr
from datetime import timedelta
from mpi4py import MPI
import parcels


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
# VALIDATE MPI / ENSEMBLE CONFIGURATION
# ================================================================

if size % N_MEMBERS != 0:
    if rank == 0:
        print(
            f"[ERROR] Number of MPI ranks ({size}) must be a multiple of "
            f"N_MEMBERS ({N_MEMBERS}). Aborting.",
            flush=True,
        )
    comm.Barrier()
    sys.exit(1)

# Number of cores working in parallel on each member
CORES_PER_MEMBER = size // N_MEMBERS

# Each rank is assigned to exactly one member (1-indexed)
member = rank // CORES_PER_MEMBER + 1

# Index of this rank within its member group (0-indexed: 0 … CORES_PER_MEMBER-1)
local_rank = rank % CORES_PER_MEMBER


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

def create_particle_cloud(lon, lat, depth, rng):
    """
    Generate a particle cloud around each release point.

    Parameters
    ----------
    lon, lat, depth : array-like
        Release point coordinates.
    rng : numpy.random.Generator
        Random number generator with a rank-specific seed so that each
        core draws a distinct random sample.

    If PARTICLES_PER_POINT > 1: 2D Gaussian distribution (lon, lat)
    centred on the release point, with standard deviation GAUSSIAN_STD_DEG
    in degrees.
    If PARTICLES_PER_POINT == 1: a single particle placed exactly on the
    release point.
    """

    lon_list = []
    lat_list = []
    depth_list = []

    for lo, la, de in zip(lon, lat, depth):
        if PARTICLES_PER_POINT > 1:
            lo_cloud, la_cloud = rng.multivariate_normal(
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
# RUN THIS RANK'S PARTICLE SUBSET FOR ONE ENSEMBLE MEMBER
# ================================================================

def run_rank(member, local_rank, lon, lat, depth):
    """
    Advect the particle subset assigned to this MPI rank.

    Each rank works on exactly one ensemble member and one
    velocity file.  The subset of particles is a random sample
    drawn with a seed that is unique per rank, so every core
    receives a different (non-overlapping) realisation of the
    Gaussian cloud.

    Output files are named:
        traj_member_{member:03d}_part{local_rank:02d}.zarr
    """

    fname = FILE_PATTERN.format(member)
    file = os.path.join(ENSEMBLE_DIR, fname)

    if not os.path.exists(file):
        print(f"[rank {rank}] Missing {file}", flush=True)
        return

    print(
        f"[rank {rank:02d}] member={member:03d} local_rank={local_rank:02d} "
        f"particles={len(lon)}",
        flush=True,
    )

    fieldset, timespan = create_fieldset(file)

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=lon, lat=lat, depth=depth
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_name = os.path.join(
        OUTPUT_DIR,
        f"traj_member_{member:03d}_part{local_rank:02d}.zarr"
    )
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

    print(
        f"[rank {rank:02d}] Finished member={member:03d} "
        f"part={local_rank:02d}",
        flush=True,
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    if rank == 0:
        print("\nMPI parallelization: multiple cores per member", flush=True)
        print(f"  MPI ranks:          {size}", flush=True)
        print(f"  Ensemble members:   {N_MEMBERS}", flush=True)
        print(f"  Cores per member:   {CORES_PER_MEMBER}", flush=True)

    # Load release points on every rank (small file, no broadcast needed)
    lon, lat, depth = load_release_points()

    # Each rank uses a unique seed so that its random particle cloud
    # is statistically independent from those of sibling ranks.
    # Seed = rank ensures reproducibility across re-runs.
    rng = np.random.default_rng(seed=rank)

    lon_p, lat_p, depth_p = create_particle_cloud(lon, lat, depth, rng)

    if rank == 0:
        print(f"\nRelease points:              {len(lon)}", flush=True)
        print(f"Particles per point:         {PARTICLES_PER_POINT}", flush=True)
        print(f"Total particles per member:  {len(lon_p)} x {CORES_PER_MEMBER} cores", flush=True)

    print(
        f"[rank {rank:02d}] assigned member={member:03d} "
        f"local_rank={local_rank:02d}",
        flush=True,
    )

    run_rank(member, local_rank, lon_p, lat_p, depth_p)

    comm.Barrier()

    if rank == 0:
        print("\nAll simulations completed.", flush=True)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
