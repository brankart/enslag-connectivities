#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mean_swiose.py
Compute the mean trajectory of each ensemble member's .zarr output
and save all mean trajectories into a single .zarr file.
"""


# ================================================================
# ENVIRONMENT
# ================================================================

import os
import xarray as xr
import matplotlib.pyplot as plt


# ================================================================
# GLOBAL PARAMETERS
# ================================================================

nens = 30
npart = 100
dpt = -10
zone = "Glorioso"
per_member_means = []


# ================================================================
# PATHS
# ================================================================

dir_bin = f"output/{zone}/z{abs(dpt)}/{npart}part"
file_partmean_bin = f"traj_{npart}part_PARTMEAN_{zone}_z{abs(dpt)}_dt100.zarr"
path_partmean_bin = os.path.join(dir_bin, file_partmean_bin)

member_files = [
    os.path.join(
        dir_bin,
        f"traj_{npart}part_memb{idx:03d}_{zone}_z{abs(dpt)}_dt100.zarr"
    )
    for idx in range(1, nens + 1)
]


# ================================================================
# PER-MEMBER MEANS
# ================================================================

print()

for f in member_files:

    print(f"Processing {f}...")

    if not os.path.exists(f):
        print(f"⚠️ Missing file: {f}, skipping.")
        continue

    ds = xr.open_zarr(f)

    # Use the first dimension name (usually "trajectory")
    first_dim = list(ds.dims)[0]

    ds_mean = ds.mean(dim=first_dim, skipna=True, keep_attrs=True)
    per_member_means.append(ds_mean)

if not per_member_means:
    raise FileNotFoundError("No member .zarr files found to average.")


# ================================================================
# CONCATENATE AND SAVE
# ================================================================

combined = xr.concat(per_member_means, dim="ensemble")
#combined.to_zarr(path_partmean_bin, mode="w")


#####################################################
# VISUALIZING DATA ##################################

print("Displaying trajectories...")

plt.figure()
ax = plt.axes()
ax.set_ylabel("Meridional drift [Latitude]")
ax.set_xlabel("Zonal drift [Longitude]")
ax.grid()
ax.scatter(combined["lon"].values[:,0], combined["lat"].values[:,0])
ax.scatter(combined["lon"].values[:,0:5], combined["lat"].values[:,0:5])
plt.title('Lagrangian drift of '
        +str(npart)+' particle / member (z='
        +str(dpt)+'m)')
plt.show()

#plt.savefig('figures/traj_'
#        +str(npart)+'part_'
#        +zone+'_z'
#        +str(abs(dpt))+'_dt100.png',
#        dpi=300, bbox_inches='tight',pad_inches=0.01)


print()
print(f"✅ Per-member mean trajectories saved to {path_partmean_bin}")




