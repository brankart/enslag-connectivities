#!/usr/bin/env python3
# ================================================================
# ANIMATE PARTICLES
# ----------------------------------------------------------------
# Reads OceanParcels trajectory zarr stores (one or several
# ensemble members) and produces an animated map of particle
# positions, one frame per output timestep.
#
# Each ensemble member is rendered in a distinct color.  When
# multiple proc*.zarr sub-directories are present inside a member
# directory, all partitions are concatenated automatically.
#
# OUTPUT
# ----------------------------------------------------------------
#   An animated GIF (default) or MP4 file saved under OUTPUT_DIR.
#
# COMMAND-LINE OPTIONS  (all optional – defaults are set below)
# ----------------------------------------------------------------
#   --traj-dir DIR          Directory containing traj_member_*.zarr
#                           stores [default: OUTPUT_DIR/EXPT]
#   --members M [M ...]     One or more member numbers to display,
#                           e.g. --members 1 2 3.  Omit to display
#                           all discovered members.
#   --region LON0 LON1 LAT0 LAT1
#                           Map extent in degrees.  Omit to use the
#                           bounding box of the particle data.
#   --dot-size S            Marker size for each particle dot
#                           [default: 1.0]
#   --output FILE           Output file path (.gif or .mp4).
#                           [default: <traj-dir>/particle_animation.gif]
#   --fps N                 Frames per second for the animation
#                           [default: 4]
#   --dpi N                 Resolution of each frame in DPI
#                           [default: 150]
#
# EXAMPLES
# ----------------------------------------------------------------
#   # All members, automatic region, saved as GIF
#   python animate_particles.py
#
#   # Members 1 and 3 only, custom region
#   python animate_particles.py --members 1 3 \
#       --region -6 10 35 47
#
#   # Save as MP4 at 8 fps
#   python animate_particles.py --output anim.mp4 --fps 8
# ================================================================


# ================================================================
# IMPORTS
# ================================================================

import os
import glob
import argparse
import warnings

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn(
        "cartopy is not installed.  Coastlines and land shading will be "
        "skipped.  Install cartopy for full map support.",
        stacklevel=2,
    )


# ================================================================
# USER PARAMETERS  (all can be overridden via CLI)
# ================================================================

from config import (
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    FIGURES_DIR,
    EXPT,
    N_MEMBERS,
    extract_member_id,
    TRAJ_GLOB_PATTERN,
)

# Default trajectory directory
DEFAULT_TRAJ_DIR = str(BASE_OUTPUT_DIR / EXPT)

# Default output file
DEFAULT_OUTPUT = os.path.join(DEFAULT_TRAJ_DIR, "particle_animation.gif")

# Default map extent [lon_min, lon_max, lat_min, lat_max]
# Set to None to auto-detect from the data.
DEFAULT_REGION = None

# Default dot size
DEFAULT_DOT_SIZE = 1.0

# Default animation speed
DEFAULT_FPS = 4

# Default figure resolution
DEFAULT_DPI = 150

# Colormap used to assign one color per member (qualitative palette)
MEMBER_CMAP = "tab10"


# ================================================================
# TRAJECTORY DISCOVERY
# ================================================================

def discover_member_stores(traj_dir, member_filter=None):
    """Return a dict mapping member_id -> list of zarr store paths.

    Handles two layouts produced by OceanParcels:
      - Flat:       traj_member_001.zarr               (single store)
      - Partitioned: traj_member_001.zarr/proc00.zarr  (MPI sub-dirs)

    Parameters
    ----------
    traj_dir : str
        Directory to search for trajectory stores.
    member_filter : list[int] | None
        If given, only include members whose 1-indexed number is in
        this list.

    Returns
    -------
    dict[str, list[str]]
        Ordered mapping: member_id (e.g. 'member_001') -> [path, ...]
    """

    pattern = os.path.join(traj_dir, TRAJ_GLOB_PATTERN)
    top_dirs = sorted(glob.glob(pattern))

    if not top_dirs:
        raise FileNotFoundError(
            f"No trajectory stores matching '{pattern}' found in '{traj_dir}'."
        )

    member_stores = {}

    for top in top_dirs:
        member_id = extract_member_id(top)

        # Optional filter
        if member_filter is not None:
            import re
            m = re.search(r"(\d+)", member_id)
            num = int(m.group(1)) if m else -1
            if num not in member_filter:
                continue

        # Check for partitioned sub-dirs (proc*.zarr)
        sub_dirs = sorted(glob.glob(os.path.join(top, "proc*.zarr")))
        if sub_dirs:
            member_stores[member_id] = sub_dirs
        else:
            # Flat store
            member_stores[member_id] = [top]

    if not member_stores:
        raise FileNotFoundError(
            "No matching member stores found (check --members filter and traj-dir)."
        )

    return dict(sorted(member_stores.items()))


# ================================================================
# DATA LOADING
# ================================================================

def _pad_to_length(arr, n_times):
    """Pad a (n_particles, t) array with NaN along axis 1 to width n_times.

    Parcels writes shorter time series for proc files whose particles were
    all deleted before the end of the run, so partitions may have different
    numbers of timesteps.  Padding with NaN makes them concatenatable and
    causes deleted-particle positions to be treated as missing in the
    animation (they are already filtered out by the finite-value mask).
    """
    if arr.shape[1] == n_times:
        return arr
    pad_width = n_times - arr.shape[1]
    padding = np.full((arr.shape[0], pad_width), np.nan, dtype=np.float64)
    return np.concatenate([arr.astype(np.float64), padding], axis=1)


def load_member_trajectories(store_paths):
    """Load and concatenate lon/lat arrays from a list of zarr stores.

    Different proc partitions may have different numbers of timesteps when
    some particles are deleted before the end of the run.  All arrays are
    padded with NaN to the maximum length before concatenation so that the
    resulting arrays have a uniform shape (n_particles, n_times_max).

    Parameters
    ----------
    store_paths : list[str]
        Paths to zarr stores that belong to the same member.

    Returns
    -------
    lon : ndarray, shape (n_particles, n_times)
    lat : ndarray, shape (n_particles, n_times)
    times : ndarray of datetime64, shape (n_times,)
        Time axis taken from the partition with the most timesteps.
    """

    lons, lats = [], []
    times = None
    n_times_max = 0

    for path in store_paths:
        with xr.open_zarr(path) as ds:
            lon_chunk = ds["lon"].values
            lat_chunk = ds["lat"].values
            lons.append(lon_chunk)
            lats.append(lat_chunk)

            n_t = lon_chunk.shape[1]

            # Keep the longest time axis as the reference
            if n_t > n_times_max:
                n_times_max = n_t
                if "time" in ds:
                    t = ds["time"].values      # (n_particles, n_times)
                    times = t[0]
                elif "obs" in ds.dims:
                    times = ds["obs"].values
                else:
                    times = np.arange(n_t)

    # Pad all partitions to the common length, then concatenate
    lons = [_pad_to_length(a, n_times_max) for a in lons]
    lats = [_pad_to_length(a, n_times_max) for a in lats]

    lon = np.concatenate(lons, axis=0)   # (total_particles, n_times_max)
    lat = np.concatenate(lats, axis=0)

    return lon, lat, times


# ================================================================
# MAP HELPERS
# ================================================================

def make_map_axes(fig, region):
    """Create a cartopy or plain axes, applying *region* as extent.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    region : list[float] | None
        [lon_min, lon_max, lat_min, lat_max] or None (auto-extent).

    Returns
    -------
    ax, transform_kwargs
    """
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.LAND,
            facecolor="lightgrey",
            edgecolor="none",
            zorder=2,
        )
        ax.add_feature(
            cfeature.COASTLINE,
            linewidth=0.7,
            edgecolor="black",
            zorder=3,
        )
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, zorder=4)
        if region is not None:
            ax.set_extent(region, crs=ccrs.PlateCarree())
        tkw = {"transform": ccrs.PlateCarree(), "zorder": 5}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect("equal")
        if region is not None:
            ax.set_xlim(region[0], region[1])
            ax.set_ylim(region[2], region[3])
        tkw = {"zorder": 5}

    return ax, tkw


def auto_region(member_data):
    """Compute a bounding box that contains all particle positions."""
    all_lons = np.concatenate([d["lon"].ravel() for d in member_data.values()])
    all_lats = np.concatenate([d["lat"].ravel() for d in member_data.values()])

    # Ignore NaN (deleted particles)
    all_lons = all_lons[np.isfinite(all_lons)]
    all_lats = all_lats[np.isfinite(all_lats)]

    margin = 0.5   # degrees of padding
    return [
        float(all_lons.min()) - margin,
        float(all_lons.max()) + margin,
        float(all_lats.min()) - margin,
        float(all_lats.max()) + margin,
    ]


def format_time_label(t):
    """Return a human-readable string for a numpy datetime64 or index."""
    if isinstance(t, (np.datetime64,)):
        # Convert to string, trim sub-second precision
        return str(t)[:16].replace("T", " ")
    return f"t = {t}"


# ================================================================
# ANIMATION
# ================================================================

def build_animation(member_data, member_ids, colors, region, dot_size, fps, dpi):
    """Build a matplotlib.animation.FuncAnimation.

    Parameters
    ----------
    member_data : dict[str, dict]
        {member_id: {"lon": ndarray, "lat": ndarray, "times": ndarray}}
    member_ids : list[str]
        Ordered list of member ids (for legend and color assignment).
    colors : list
        One color per member.
    region : list[float]
        [lon_min, lon_max, lat_min, lat_max]
    dot_size, fps, dpi : as per CLI args.

    Returns
    -------
    fig, anim
    """

    # All members must share the same time axis (taken from member_data)
    times = next(iter(member_data.values()))["times"]
    n_times = len(times)

    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    ax, tkw = make_map_axes(fig, region)

    # Create one scatter artist per member (initially empty)
    scatters = {}
    for mid, color in zip(member_ids, colors):
        sc = ax.scatter(
            [], [],
            s=dot_size,
            c=[color],
            label=mid,
            alpha=0.6,
            linewidths=0,
            **tkw,
        )
        scatters[mid] = sc

    # Legend (only if multiple members)
    if len(member_ids) > 1:
        ax.legend(
            loc="upper right",
            fontsize=7,
            markerscale=4,
            framealpha=0.8,
        )

    title = ax.set_title("", fontsize=10)

    def init():
        for sc in scatters.values():
            sc.set_offsets(np.empty((0, 2)))
        title.set_text("")
        return list(scatters.values()) + [title]

    def update(frame_idx):
        for mid in member_ids:
            d = member_data[mid]
            lon_t = d["lon"][:, frame_idx]
            lat_t = d["lat"][:, frame_idx]
            # Mask deleted/out-of-domain particles
            mask = np.isfinite(lon_t) & np.isfinite(lat_t)
            if mask.any():
                offsets = np.column_stack([lon_t[mask], lat_t[mask]])
            else:
                offsets = np.empty((0, 2))
            scatters[mid].set_offsets(offsets)

        title.set_text(f"Particle positions — {format_time_label(times[frame_idx])}")
        return list(scatters.values()) + [title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_times,
        init_func=init,
        blit=True,
        interval=int(1000 / fps),
    )

    return fig, anim


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Animate OceanParcels particle trajectories for one or "
            "several ensemble members."
        )
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default=DEFAULT_TRAJ_DIR,
        metavar="DIR",
        help=(
            f"Directory containing traj_member_*.zarr stores "
            f"[default: {DEFAULT_TRAJ_DIR}]"
        ),
    )
    parser.add_argument(
        "--members",
        type=int,
        nargs="+",
        default=None,
        metavar="M",
        help=(
            "One or more member numbers to display (e.g. --members 1 2 3). "
            "Omit to display all discovered members."
        ),
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help=(
            "Map extent in degrees: lon_min lon_max lat_min lat_max. "
            "Omit to auto-detect from particle data."
        ),
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=DEFAULT_DOT_SIZE,
        metavar="S",
        help=f"Marker size for particle dots [default: {DEFAULT_DOT_SIZE}]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=(
            f"Output file path (.gif or .mp4) "
            f"[default: {DEFAULT_OUTPUT}]"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        metavar="N",
        help=f"Frames per second [default: {DEFAULT_FPS}]",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="N",
        help=f"Figure resolution in DPI [default: {DEFAULT_DPI}]",
    )
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():
    args = parse_args()

    # ----------------------------------------------------------
    # 1. Discover trajectory stores
    # ----------------------------------------------------------
    print(f"Scanning trajectory directory: {args.traj_dir}", flush=True)
    member_stores = discover_member_stores(args.traj_dir, member_filter=args.members)
    member_ids = list(member_stores.keys())
    print(f"Found {len(member_ids)} member(s): {', '.join(member_ids)}", flush=True)

    # ----------------------------------------------------------
    # 2. Load lon / lat for each member
    # ----------------------------------------------------------
    member_data = {}
    for mid, paths in member_stores.items():
        print(f"  Loading {mid} ({len(paths)} partition(s)) ...", flush=True)
        lon, lat, times = load_member_trajectories(paths)
        member_data[mid] = {"lon": lon, "lat": lat, "times": times}
        print(
            f"    → {lon.shape[0]} particles × {lon.shape[1]} timesteps",
            flush=True,
        )

    # ----------------------------------------------------------
    # 3. Determine map region
    # ----------------------------------------------------------
    region = args.region if args.region is not None else auto_region(member_data)
    print(
        f"Map region: lon [{region[0]:.2f}, {region[1]:.2f}]  "
        f"lat [{region[2]:.2f}, {region[3]:.2f}]",
        flush=True,
    )

    # ----------------------------------------------------------
    # 4. Assign one color per member
    # ----------------------------------------------------------
    cmap = plt.get_cmap(MEMBER_CMAP)
    n = max(len(member_ids), 1)
    colors = [cmap(i / n) for i in range(n)]

    # ----------------------------------------------------------
    # 5. Build animation
    # ----------------------------------------------------------
    print("Building animation ...", flush=True)
    fig, anim = build_animation(
        member_data,
        member_ids,
        colors,
        region,
        dot_size=args.dot_size,
        fps=args.fps,
        dpi=args.dpi,
    )

    # ----------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
    else:
        # Default to GIF (pillow writer)
        writer = animation.PillowWriter(fps=args.fps)

    print(f"Saving animation to: {out_path}", flush=True)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print("Done.", flush=True)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
