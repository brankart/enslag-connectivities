#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getvars_swiose.py
Retrieves scalar fields from CROCO netCDF files.

"""

# ================================================================
# ENVIRONMENT
# ================================================================

import os
import sys
import errno
import numpy as np
import xarray as xr
import parcels
import tools as tl
from config import (
    ENSEMBLE_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    N_MEMBERS,
)


# ================================================================
# SETTINGS
# ================================================================

# idx = 1 # member index
ensemble = range(1, N_MEMBERS + 1)
sample = ensemble
override = True # set to True to overwrite existing file
year = '2017'
ens_id = 'ini'
file_type = "avg"
var2get = ['ssh']

swio_dataset_folder = os.environ.get(
    "ENSLAG_DATA_DIR",
    f"{ENSEMBLE_DIR}_{ens_id}" if str(ENSEMBLE_DIR) != "your_path_to_ensemble_data" else f"your_path_to_ensemble_data_{ens_id}"
)
fname = [f"{idx:03d}swiose_{file_type}.nc" for idx in ensemble]
file = [os.path.join(swio_dataset_folder, f) for f in fname]

file_bathy_bin = str(OUTPUT_DIR / ens_id / "bathy.zarr")
file_bathy_fig = str(FIGURES_DIR / ens_id / "bathy.png")
file_ssh_bin = [str(OUTPUT_DIR / ens_id / "modelvars/ssh" / f"ssh_memb{idx:03d}_{year}_{ens_id}.zarr") for idx in ensemble]
file_ssh_fig = [str(FIGURES_DIR / ens_id / "modelvars/ssh" / f"ssh_memb{idx:03d}_{year}_{ens_id}.png") for idx in ensemble]


# ================================================================
# EXTRACTION
# ================================================================

if override and 'bathy' in var2get :
    tl.get_bathy(file, file_bathy_bin, force_reextract=True)
elif 'bathy' in var2get :
    tl.get_bathy(file, file_bathy_bin)

if override and 'ssh' in var2get :
    tl.get_ssh(file, file_ssh_bin, force_reextract=True)
elif 'ssh' in var2get :
    tl.get_ssh(file, file_ssh_bin)





