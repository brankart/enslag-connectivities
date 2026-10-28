#!/usr/bin/env python3
"""
Configuration module for enslag-connectivities.

Dynamically resolves project directories and centralizes shared
parameters for simulation, connectivity analysis, and visualization,
with support for environment variable overrides.
"""

import os
import re
import sys
from pathlib import Path
from typing import Union

# ================================================================
# PROJECT DIRECTORIES (dynamically resolved)
# ================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core project subdirectories
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src/ is in sys.path for direct imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ================================================================
# DATA PATHS & ENVIRONMENT VARIABLE OVERRIDES
# ================================================================

# Path to CROCO ensemble netCDF output directory
# Override with environment variable ENSLAG_DATA_DIR or ENSEMBLE_DIR
DEFAULT_ENSEMBLE_DIR = "your_path_to_ensemble_data"
ENSEMBLE_DIR = Path(
    os.environ.get("ENSLAG_DATA_DIR", os.environ.get("ENSEMBLE_DIR", DEFAULT_ENSEMBLE_DIR))
)

# Release spots definition file
DEFAULT_SPOTS_FILE = INPUT_DIR / "release_spots.txt"
SPOTS_FILE = Path(os.environ.get("ENSLAG_SPOTS_FILE", DEFAULT_SPOTS_FILE))


# ================================================================
# SHARED EXPERIMENT & SIMULATION PARAMETERS
# ================================================================

# Ensemble size
N_MEMBERS = int(os.environ.get("ENSLAG_N_MEMBERS", 30))

# Particles per seeding spot
PARTICLES_PER_POINT = int(os.environ.get("ENSLAG_PARTICLES_PER_POINT", 4000))

# Standard deviation of 2D Gaussian seeding cloud in degrees
GAUSSIAN_STD_DEG = float(os.environ.get("ENSLAG_GAUSSIAN_STD_DEG", 0.05))

# Simulation advection timestep in seconds
DT = int(os.environ.get("ENSLAG_DT", 900))


# ================================================================
# CONNECTIVITY & DISCRETIZATION PARAMETERS
# ================================================================

# Side length of square subdomain boxes in km
BOX_SIZE_KM = float(os.environ.get("ENSLAG_BOX_SIZE_KM", 50.0))

# Site arrival radius in km for connectivity network
SITE_RADIUS_KM = float(os.environ.get("ENSLAG_SITE_RADIUS_KM", 25.0))

# Temporal binning window in days
TIME_BIN_DAYS = float(os.environ.get("ENSLAG_TIME_BIN_DAYS", 5.0))


# ================================================================
# CONVENIENCE HELPERS & MEMBER IDENTIFIER STANDARDIZATION
# ================================================================

TRAJ_FILENAME_PATTERN = "traj_member_{:03d}.zarr"
TRAJ_GLOB_PATTERN = "traj_member_*.zarr"


def format_member_id(member: Union[int, str]) -> str:
    """Standardize an ensemble member identifier to 'member_XXX' (e.g. 1 -> 'member_001')."""
    m = re.search(r"(\d+)", str(member))
    if m:
        return f"member_{int(m.group(1)):03d}"
    return str(member)


def extract_member_id(path: Union[str, Path]) -> str:
    """Extract standardized member identifier 'member_XXX' from a file or directory path.

    Matches both flat trajectory stores (e.g. traj_member_001.zarr) and
    partitioned MPI rank subdirectories (e.g. traj_member_001.zarr/proc00.zarr).
    """
    m = re.search(r"traj_member_(\d+)", str(path))
    if m:
        return f"member_{int(m.group(1)):03d}"
    # Fallback to general numeric search or basename
    return format_member_id(Path(path).name.replace(".zarr", ""))


def get_traj_dir(ens_id: str = "ini") -> Path:
    """Return the output directory path for a given experiment ID."""
    return OUTPUT_DIR / ens_id


def get_figures_dir(ens_id: str = "ini") -> Path:
    """Return the figures directory path for a given experiment ID."""
    return FIGURES_DIR / ens_id
