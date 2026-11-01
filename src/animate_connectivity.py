#!/usr/bin/env python3
# ================================================================
# ANIMATE CONNECTIVITY
# ----------------------------------------------------------------
# Reads OceanParcels trajectory zarr stores (one or several
# ensemble members) and produces an animated connectivity map
# showing the fraction of particles that have come within a given
# radius of each grid pixel, cumulated from t=0 up to each frame.
#
# The script is self-contained and does NOT depend on config.py.
#
# OUTPUT
# ----------------------------------------------------------------
#   One animated GIF (default) or MP4 per requested member,
#   saved in the trajectory directory (or --output-dir).
#
# COMMAND-LINE OPTIONS  (all optional – defaults are set below)
# ----------------------------------------------------------------
#   --traj-dir DIR          Directory containing traj_member_*.zarr
#                           stores.
#                           [default: current working directory]
#   --members M [M ...]     Member numbers to animate, e.g.
#                           --members 1 2 3.  Omit for all members.
#   --radius KM             Closeness threshold in km: a particle
#                           contributes to a pixel if it passes
#                           within this distance of the pixel centre.
#                           [default: 25 km]
#   --grid-deg DEG          Resolution of the connectivity grid in
#                           degrees.  [default: 0.1°]
#   --region LON0 LON1 LAT0 LAT1
#                           Map extent in degrees.  Omit to
#                           auto-detect from the particle data.
#   --cmap NAME             Matplotlib colormap for connectivity
#                           (applied in log scale, white for zero).
#                           [default: plasma]
#   --output-dir DIR        Directory for output files.
#                           [default: same as --traj-dir]
#   --format {gif,mp4}      Output file format: 'gif' (Pillow writer,
#                           default) or 'mp4' (FFMpeg; must be installed).
#                           [default: gif]
#   --fps N                 Frames per second.  [default: 4]
#   --dpi N                 Figure resolution in DPI.  [default: 150]
#
# EXAMPLES
# ----------------------------------------------------------------
#   # All members, 25 km radius, auto region
#   python animate_connectivity.py --traj-dir /path/to/output/expt
#
#   # Members 1 and 2, 50 km radius, Mediterranean extent
#   python animate_connectivity.py \
#       --traj-dir /path/to/output/expt \
#       --members 1 2 \
#       --radius 50 \
#       --region -6 10 35 47
#
#   # Save as MP4
#   python animate_connectivity.py \
#       --traj-dir /path/to/output/expt \
#       --output-dir /path/to/figures \
#       --fps 6
# ================================================================


# ================================================================
# IMPORTS
# ================================================================

import os
import glob
import re
import argparse
import warnings

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")               # non-interactive backend
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
# DEFAULT PARAMETERS
# ================================================================

DEFAULT_TRAJ_DIR   = os.getcwd()
DEFAULT_RADIUS_KM  = 25.0          # closeness threshold in km
DEFAULT_GRID_DEG   = 0.1           # grid resolution in degrees
DEFAULT_REGION     = None          # auto-detect
DEFAULT_CMAP       = "plasma"
DEFAULT_OUTPUT_DIR = None          # same as traj-dir
DEFAULT_FPS        = 4
DEFAULT_DPI        = 150

# Glob pattern for trajectory top-level directories
TRAJ_GLOB = "traj_member_*.zarr"
MEMBER_RE  = re.compile(r"traj_member_(\d+)")


# ================================================================
# HAVERSINE DISTANCE (vectorised)
# ================================================================

EARTH_RADIUS_KM = 6371.0

def haversine_km(lon1, lat1, lon2, lat2):
    """Return the great-circle distance in km between two points.

    Parameters
    ----------
    lon1, lat1 : float or ndarray  (degrees)
    lon2, lat2 : float or ndarray  (degrees)
        Arrays must be broadcastable against each other.

    Returns
    -------
    ndarray of distances in km.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ================================================================
# TRAJECTORY DISCOVERY
# ================================================================

def discover_member_stores(traj_dir, member_filter=None):
    """Return a dict mapping member_id (str) -> list of zarr store paths.

    Handles two layouts produced by OceanParcels:
      - Flat:        traj_member_001.zarr
      - Partitioned: traj_member_001.zarr/proc00.zarr, proc01.zarr, …

    Parameters
    ----------
    traj_dir : str
    member_filter : list[int] | None
        If given, only include members whose 1-indexed number is in
        this list.
    """

    top_dirs = sorted(glob.glob(os.path.join(traj_dir, TRAJ_GLOB)))

    if not top_dirs:
        raise FileNotFoundError(
            f"No trajectory stores matching '{TRAJ_GLOB}' found in "
            f"'{traj_dir}'."
        )

    member_stores = {}

    for top in top_dirs:
        m = MEMBER_RE.search(os.path.basename(top))
        if not m:
            continue
        num = int(m.group(1))
        member_id = f"member_{num:03d}"

        if member_filter is not None and num not in member_filter:
            continue

        sub_dirs = sorted(glob.glob(os.path.join(top, "proc*.zarr")))
        member_stores[member_id] = sub_dirs if sub_dirs else [top]

    if not member_stores:
        raise FileNotFoundError(
            "No matching member stores found "
            "(check --members filter and --traj-dir)."
        )

    return dict(sorted(member_stores.items()))


# ================================================================
# DATA LOADING
# ================================================================

def _pad_to_length(arr, n_times):
    """Pad a (n_particles, t) float array with NaN along axis 1."""
    if arr.shape[1] == n_times:
        return arr
    pad = np.full(
        (arr.shape[0], n_times - arr.shape[1]), np.nan, dtype=np.float64
    )
    return np.concatenate([arr.astype(np.float64), pad], axis=1)


def load_member_trajectories(store_paths):
    """Load and concatenate lon/lat from a list of zarr stores.

    Partitions with shorter time axes (due to particle deletion) are
    padded with NaN before concatenation.

    Returns
    -------
    lon : ndarray (n_particles, n_times)
    lat : ndarray (n_particles, n_times)
    times : ndarray (n_times,)  datetime64 or integer indices
    """

    lons, lats = [], []
    times = None
    n_times_max = 0

    for path in store_paths:
        with xr.open_zarr(path) as ds:
            lon_c = ds["lon"].values
            lat_c = ds["lat"].values
            lons.append(lon_c)
            lats.append(lat_c)

            n_t = lon_c.shape[1]
            if n_t > n_times_max:
                n_times_max = n_t
                if "time" in ds:
                    times = ds["time"].values[0]   # first particle's axis
                elif "obs" in ds.dims:
                    times = ds["obs"].values
                else:
                    times = np.arange(n_t)

    lons = [_pad_to_length(a, n_times_max) for a in lons]
    lats = [_pad_to_length(a, n_times_max) for a in lats]

    return (
        np.concatenate(lons, axis=0),
        np.concatenate(lats, axis=0),
        times,
    )


# ================================================================
# GRID CONSTRUCTION
# ================================================================

def build_grid(region, grid_deg):
    """Build a regular lon/lat grid over *region*.

    Returns
    -------
    lon_grid : 1-D ndarray (n_lon,)
    lat_grid : 1-D ndarray (n_lat,)
    LON, LAT : 2-D meshgrid arrays (n_lat, n_lon)
    """
    lon_grid = np.arange(region[0], region[1] + 1e-9, grid_deg)
    lat_grid = np.arange(region[2], region[3] + 1e-9, grid_deg)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    return lon_grid, lat_grid, LON, LAT


def auto_region(lon, lat, margin=0.5):
    """Bounding box of finite particle positions with a margin."""
    fin_lon = lon[np.isfinite(lon)]
    fin_lat = lat[np.isfinite(lat)]
    return [
        float(fin_lon.min()) - margin,
        float(fin_lon.max()) + margin,
        float(fin_lat.min()) - margin,
        float(fin_lat.max()) + margin,
    ]


# ================================================================
# CONNECTIVITY COMPUTATION
# ================================================================

def compute_cumulative_connectivity(lon, lat, LON, LAT, radius_km):
    """Compute cumulative connectivity grids for each timestep.

    For each timestep t, the connectivity grid C[t] gives the fraction
    of (living) particles that have come within *radius_km* of each
    grid pixel at any point from t=0 up to t.

    Parameters
    ----------
    lon, lat : ndarray (n_particles, n_times)
    LON, LAT : ndarray (n_lat, n_lon)  – grid centres
    radius_km : float

    Returns
    -------
    connectivity : ndarray (n_times, n_lat, n_lon)
        Values in [0, 1].  NaN where no particles were ever released
        (all-NaN slice).
    n_total_particles : int
        Total number of particles (including those deleted during run).
    """

    n_particles, n_times = lon.shape
    n_lat, n_lon = LON.shape

    # Boolean mask: has particle p ever visited pixel (i,j) by time t?
    # We build it cumulatively to avoid storing the full (p, t, i, j) array.
    ever_visited = np.zeros((n_particles, n_lat, n_lon), dtype=bool)

    # Pre-flatten the grid for vectorised haversine
    grid_lon_flat = LON.ravel()    # (n_lat * n_lon,)
    grid_lat_flat = LAT.ravel()

    connectivity = np.full((n_times, n_lat, n_lon), np.nan, dtype=np.float64)

    for t in range(n_times):
        lon_t = lon[:, t]    # (n_particles,)
        lat_t = lat[:, t]

        # Alive particles at this timestep
        alive = np.isfinite(lon_t) & np.isfinite(lat_t)

        if alive.any():
            # For each alive particle, find pixels within radius_km.
            # Broadcast: (n_alive, 1) vs (1, n_pixels)
            lon_alive = lon_t[alive][:, np.newaxis]    # (n_alive, 1)
            lat_alive = lat_t[alive][:, np.newaxis]

            dist = haversine_km(
                lon_alive, lat_alive,
                grid_lon_flat[np.newaxis, :],
                grid_lat_flat[np.newaxis, :],
            )   # (n_alive, n_pixels)

            within = dist <= radius_km               # (n_alive, n_pixels)
            within_2d = within.reshape(
                lon_alive.shape[0], n_lat, n_lon
            )                                        # (n_alive, n_lat, n_lon)

            # Update ever_visited for alive particles
            ever_visited[alive] |= within_2d

        # Connectivity fraction: particles that ever visited / total
        connectivity[t] = ever_visited.sum(axis=0) / n_particles

    return connectivity, n_particles


# ================================================================
# MAP HELPERS
# ================================================================

def make_map_axes(fig, region):
    """Create a cartopy or plain axes, applying *region* as extent."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.LAND, facecolor="lightgrey", edgecolor="none", zorder=2
        )
        ax.add_feature(
            cfeature.COASTLINE,
            linewidth=0.7, edgecolor="black", zorder=3
        )
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, zorder=4)
        ax.set_extent(region, crs=ccrs.PlateCarree())
        tkw = {"transform": ccrs.PlateCarree(), "zorder": 1}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect("equal")
        ax.set_xlim(region[0], region[1])
        ax.set_ylim(region[2], region[3])
        tkw = {"zorder": 1}

    return ax, tkw


def format_time_label(t):
    """Human-readable string for a numpy datetime64 or index."""
    if isinstance(t, np.datetime64):
        return str(t)[:16].replace("T", " ")
    return f"step {t}"


# ================================================================
# BUILD ANIMATION FOR ONE MEMBER
# ================================================================

def build_member_animation(
    member_id, connectivity, times, lon_grid, lat_grid, region,
    cmap, fps, dpi
):
    """Build a FuncAnimation for one member's connectivity sequence.

    Parameters
    ----------
    connectivity : ndarray (n_times, n_lat, n_lon)
    times : ndarray (n_times,)
    lon_grid, lat_grid : 1-D coordinate arrays
    region : [lon0, lon1, lat0, lat1]
    cmap, fps, dpi : display parameters

    Returns
    -------
    fig, anim
    """

    n_times = connectivity.shape[0]

    # Build a colormap with white for zero / NaN
    base_cmap = plt.get_cmap(cmap)
    cmap_obj = base_cmap.copy()
    cmap_obj.set_bad(color="white")     # NaN → white
    cmap_obj.set_under(color="white")   # below vmin → white

    # Log normalization: vmin just above zero to keep zero cells white
    vmax = float(np.nanmax(connectivity))
    vmax = vmax if vmax > 0 else 1.0
    vmin = max(float(np.nanmin(connectivity[connectivity > 0])) if np.any(connectivity > 0) else 1e-4,
               vmax * 1e-4)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # Mask zero values so they appear white
    def mask_zeros(grid):
        out = grid.copy().astype(np.float64)
        out[out <= 0] = np.nan
        return out

    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    ax, tkw = make_map_axes(fig, region)

    # Initial frame
    init_data = mask_zeros(connectivity[0])
    img = ax.pcolormesh(
        lon_grid, lat_grid, init_data,
        cmap=cmap_obj, norm=norm,
        shading="auto",
        **tkw,
    )

    cbar = fig.colorbar(img, ax=ax, shrink=0.8, pad=0.02, extend="neither")
    cbar.set_label("Fraction of particles (cumulative)")

    title = ax.set_title(
        f"{member_id} — connectivity — {format_time_label(times[0])}",
        fontsize=10,
    )
    fig.tight_layout()

    def init():
        img.set_array(mask_zeros(connectivity[0]).ravel())
        title.set_text(
            f"{member_id} — connectivity — {format_time_label(times[0])}"
        )
        return [img, title]

    def update(frame_idx):
        img.set_array(mask_zeros(connectivity[frame_idx]).ravel())
        title.set_text(
            f"{member_id} — connectivity — "
            f"{format_time_label(times[frame_idx])}"
        )
        return [img, title]

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
            "Animate cumulative Lagrangian connectivity maps from "
            "OceanParcels ensemble trajectory stores."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--traj-dir",
        default=DEFAULT_TRAJ_DIR,
        metavar="DIR",
        help="Directory containing traj_member_*.zarr stores.",
    )
    parser.add_argument(
        "--members",
        type=int, nargs="+", default=None, metavar="M",
        help=(
            "Member numbers to animate (e.g. --members 1 2 3). "
            "Omit for all discovered members."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float, default=DEFAULT_RADIUS_KM, metavar="KM",
        help="Closeness threshold in km.",
    )
    parser.add_argument(
        "--grid-deg",
        type=float, default=DEFAULT_GRID_DEG, metavar="DEG",
        help="Grid resolution in degrees.",
    )
    parser.add_argument(
        "--region",
        type=float, nargs=4, default=None,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help=(
            "Map extent: lon_min lon_max lat_min lat_max. "
            "Omit to auto-detect from particle data."
        ),
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP, metavar="NAME",
        help="Matplotlib colormap name (log scale, white for zero).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=(
            "Directory for output animation files. "
            "Defaults to --traj-dir."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"], default="gif",
        help=(
            "Output file format: 'gif' (Pillow writer, default) or "
            "'mp4' (FFMpeg writer; FFmpeg must be installed)."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int, default=DEFAULT_FPS, metavar="N",
        help="Frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int, default=DEFAULT_DPI, metavar="N",
        help="Figure resolution in DPI.",
    )
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.traj_dir
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Discover stores
    # ----------------------------------------------------------
    print(f"\nScanning trajectory directory: {args.traj_dir}", flush=True)
    member_stores = discover_member_stores(
        args.traj_dir, member_filter=args.members
    )
    member_ids = list(member_stores.keys())
    print(f"Found {len(member_ids)} member(s): {', '.join(member_ids)}", flush=True)

    # ----------------------------------------------------------
    # 2. Process each member independently
    # ----------------------------------------------------------
    for mid, paths in member_stores.items():
        print(f"\n--- {mid} ({len(paths)} partition(s)) ---", flush=True)

        # 2a. Load trajectories
        print("  Loading trajectories ...", flush=True)
        lon, lat, times = load_member_trajectories(paths)
        n_particles, n_times = lon.shape
        print(
            f"  {n_particles} particles × {n_times} timesteps",
            flush=True,
        )

        # 2b. Determine grid region
        if args.region is not None:
            region = args.region
        else:
            region = auto_region(lon, lat)
        print(
            f"  Region: lon [{region[0]:.2f}, {region[1]:.2f}]  "
            f"lat [{region[2]:.2f}, {region[3]:.2f}]",
            flush=True,
        )

        # 2c. Build grid
        lon_grid, lat_grid, LON, LAT = build_grid(region, args.grid_deg)
        print(
            f"  Grid: {len(lat_grid)} × {len(lon_grid)} pixels "
            f"at {args.grid_deg}° resolution",
            flush=True,
        )

        # 2d. Compute cumulative connectivity
        print(
            f"  Computing connectivity (radius = {args.radius} km) ...",
            flush=True,
        )
        connectivity, _ = compute_cumulative_connectivity(
            lon, lat, LON, LAT, args.radius
        )

        # 2e. Build animation
        print("  Building animation ...", flush=True)
        fig, anim = build_member_animation(
            mid, connectivity, times,
            lon_grid, lat_grid, region,
            cmap=args.cmap,
            fps=args.fps,
            dpi=args.dpi,
        )

        # 2f. Save
        out_path = os.path.join(output_dir, f"connectivity_{mid}.{args.format}")
        writer = (
            animation.FFMpegWriter(fps=args.fps, bitrate=1800)
            if args.format == "mp4"
            else animation.PillowWriter(fps=args.fps)
        )
        print(f"  Saving → {out_path}", flush=True)
        anim.save(out_path, writer=writer)
        plt.close(fig)
        print(f"  Done.", flush=True)

    print("\nAll members processed.", flush=True)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
