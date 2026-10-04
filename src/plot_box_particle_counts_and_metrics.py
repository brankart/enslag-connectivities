#!/usr/bin/env python3
# ================================================================
# PLOT BOX PARTICLE COUNTS
# ----------------------------------------------------------------
# Reads the box_particle_counts.nc / box_passages_long.csv files
# produced by count_particles_per_box.py and plots a colormap of
# the number of particles that made at least one pass through each
# subdomain box.
#
# COMMAND-LINE OPTIONS (all independent, usable alone or combined)
# ----------------------------------------------------------------
#   --member MEMBER   restrict to a single ensemble member
#                      (e.g. --member 003)
#   --spot SPOT_ID     restrict to a single spawning point
#                      (e.g. --spot 4); requires box_passages_long.csv
#                      generated with GROUP_BY_SPAWN=True, and its
#                      'ilat'/'ilon' columns (added by the current
#                      version of count_particles_per_box.py)
#   --save             save the figure(s) as .png files under
#                      OUTPUT_DIR instead of displaying them
#
# Examples:
#   plot_box_particle_counts.py                        -> show total map
#   plot_box_particle_counts.py --member 003            -> show member 003, all spots
#   plot_box_particle_counts.py --spot 4                -> show spot 4, all members
#   plot_box_particle_counts.py --member 003 --spot 4   -> show member 003 / spot 4 only
#   plot_box_particle_counts.py --save                  -> save total map to OUTPUT_DIR
#   plot_box_particle_counts.py --member 003 --save     -> save member 003 map
#
# With none of --member/--spot given, the script falls back to the
# PLOT_TOTAL / PLOT_PER_MEMBER toggles below.
# ================================================================

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ================================================================
# USER PARAMETERS
# ================================================================

PATH_ABS = 'your_absolute_path_to_enslag_connectivities'
INPUT_NC = os.path.join(PATH_ABS, "output/box_particle_counts.nc")
INPUT_LONG_CSV = os.path.join(PATH_ABS, "output/box_passages_long.csv")
SPOTS_FILE = os.path.join(PATH_ABS, "input/release_spots.txt")

OUTPUT_DIR = os.path.join(PATH_ABS, "figures/ini")

PLOT_TOTAL = True            # (no-flags fallback) plot the all-members-combined map
PLOT_PER_MEMBER = False      # (no-flags fallback) also plot one map per ensemble member

LOG_SCALE = True             # counts often span orders of magnitude -> log colorbar
CMAP = "viridis"
SHOW_RELEASE_POINTS = True   # overlay release_spots.txt locations as markers

DPI = 200


# ================================================================
# HELPERS
# ================================================================

def load_release_points():
    if not os.path.exists(SPOTS_FILE):
        return None, None
    data = np.loadtxt(SPOTS_FILE, comments='#')
    return data[:, 0], data[:, 1]


def make_axes():
    """Create a figure/axes, using cartopy (with coastlines) if available,
    otherwise falling back to a plain matplotlib axes."""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        transform_kwargs = {"transform": ccrs.PlateCarree()}
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        transform_kwargs = {}

    return fig, ax, transform_kwargs


def plot_grid(lon_box, lat_box, counts, title, out_path, save,
              spawn_lon=None, spawn_lat=None):
    """Draw one colormap. If save is True, write out_path as a .png;
    otherwise display it interactively."""

    fig, ax, tkw = make_axes()

    counts_plot = counts.astype(float)
    counts_plot[counts_plot == 0] = np.nan  # leave empty boxes blank

    if LOG_SCALE:
        finite_max = np.nanmax(counts_plot) if np.any(np.isfinite(counts_plot)) else 1
        norm = mcolors.LogNorm(vmin=1, vmax=max(finite_max, 1))
    else:
        norm = None

    # box edges from box centers (assumes regular grid, as built by
    # count_particles_per_box.py)
    dlon = np.median(np.diff(lon_box))
    dlat = np.median(np.diff(lat_box))
    lon_edges = np.concatenate([lon_box - dlon / 2, [lon_box[-1] + dlon / 2]])
    lat_edges = np.concatenate([lat_box - dlat / 2, [lat_box[-1] + dlat / 2]])

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, counts_plot,
        cmap=CMAP, norm=norm, shading="flat", zorder=1, **tkw
    )

    if SHOW_RELEASE_POINTS and spawn_lon is not None:
        ax.scatter(spawn_lon, spawn_lat, s=8, c="red", marker="x",
                   linewidths=0.8, label="release points", zorder=4, **tkw)
        ax.legend(loc="upper right", fontsize=8)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Number of particles passing through box")

    ax.set_title(title)
    fig.tight_layout()

    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)
    else:
        plt.show()
        plt.close(fig)


def spot_grid_from_long_table(long_csv, spot_id, nlat, nlon, member=None):
    """Build a (nlat, nlon) count grid for a single spawning point from the
    long-format passages table, optionally restricted to a single member.

    Uses the exact integer box indices ('ilat', 'ilon') written by
    count_particles_per_box.py, rather than matching on floating-point
    box-center coordinates -- this avoids precision-related mismatches
    and is vectorised (no python-level row loop).

    NOTE: if the table was generated with GROUP_BY_TIME=True, it contains
    one row per time bin per box. Those are summed here to get a whole-run
    total, which can slightly OVER-count particles that revisit the same
    box in more than one time window. For an exact whole-run total with
    no time breakdown, regenerate the table with GROUP_BY_TIME=False.
    """

    df = pd.read_csv(long_csv)

    if 'spawn_id' not in df.columns:
        raise ValueError(
            f"{long_csv} has no 'spawn_id' column -- it must be regenerated "
            f"with GROUP_BY_SPAWN=True in count_particles_per_box.py"
        )
    if 'ilat' not in df.columns or 'ilon' not in df.columns:
        raise ValueError(
            f"{long_csv} has no 'ilat'/'ilon' columns -- it must be "
            f"regenerated with the current version of count_particles_per_box.py"
        )

    df = df[df['spawn_id'] == spot_id]

    if member is not None:
        df = df[df['member'].astype(str) == str(member)]

    if df.empty:
        raise ValueError(
            f"No rows found for spot {spot_id}"
            + (f" and member {member}" if member is not None else "")
        )

    if 'time_bin' in df.columns and df['time_bin'].nunique() > 1:
        print("NOTE: summing across time bins for this spot/member -- this "
              "can slightly over-count particles that revisit the same box "
              "in multiple time windows.", flush=True)

    grouped = df.groupby(['ilat', 'ilon'])['n_particles'].sum().reset_index()

    grid = np.zeros((nlat, nlon), dtype=np.float64)
    grid[grouped['ilat'].to_numpy(), grouped['ilon'].to_numpy()] = grouped['n_particles'].to_numpy()

    return grid


# ================================================================
# CLI ARGUMENTS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a colormap of particle passage counts per box."
    )
    parser.add_argument(
        "--member", type=str, default=None,
        help="Restrict to a single ensemble member (e.g. --member 003)."
    )
    parser.add_argument(
        "--spot", type=int, default=None,
        help="Restrict to a single spawning point index (e.g. --spot 4). "
             "Requires box_passages_long.csv generated with GROUP_BY_SPAWN=True."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the figure(s) as .png under OUTPUT_DIR instead of displaying them."
    )
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    ds = xr.open_dataset(INPUT_NC)

    lon_box = ds["lon_box"].values
    lat_box = ds["lat_box"].values
    nlat, nlon = len(lat_box), len(lon_box)

    spawn_lon, spawn_lat = load_release_points() if SHOW_RELEASE_POINTS else (None, None)

    # ------------------------------------------------------------
    # single spawning point, optionally + single member
    # (--spot alone, or --spot together with --member)
    # ------------------------------------------------------------
    if args.spot is not None:
        counts = spot_grid_from_long_table(
            INPUT_LONG_CSV, args.spot, nlat, nlon, member=args.member
        )
        title = f"Particles passing through each box -- spawn point {args.spot}"
        fname = f"box_particle_counts_spot_{args.spot}"
        if args.member is not None:
            title += f", member {args.member}"
            fname += f"_member_{args.member}"
        plot_grid(
            lon_box, lat_box, counts,
            title=title,
            out_path=os.path.join(OUTPUT_DIR, f"{fname}.png"),
            save=args.save,
            spawn_lon=spawn_lon, spawn_lat=spawn_lat
        )
        ds.close()
        return

    # ------------------------------------------------------------
    # single member, all spawning points combined (--member alone)
    # ------------------------------------------------------------
    if args.member is not None:
        counts = ds["n_particles_per_member"].sel(member=args.member).values
        plot_grid(
            lon_box, lat_box, counts,
            title=f"Number of particles passing through each box (member {args.member})",
            out_path=os.path.join(OUTPUT_DIR, f"box_particle_counts_member_{args.member}.png"),
            save=args.save,
            spawn_lon=spawn_lon, spawn_lat=spawn_lat
        )
        ds.close()
        return

    # ------------------------------------------------------------
    # default behaviour (no --member / --spot): USER PARAMETERS toggles
    # ------------------------------------------------------------
    if PLOT_TOTAL:
        counts = ds["n_particles_total"].values
        plot_grid(
            lon_box, lat_box, counts,
            title="Number of particles passing through each box (all members)",
            out_path=os.path.join(OUTPUT_DIR, "box_particle_counts_total.png"),
            save=args.save,
            spawn_lon=spawn_lon, spawn_lat=spawn_lat
        )

    if PLOT_PER_MEMBER:
        members = ds["member"].values
        for m in members:
            counts = ds["n_particles_per_member"].sel(member=m).values
            plot_grid(
                lon_box, lat_box, counts,
                title=f"Number of particles passing through each box (member {m})",
                out_path=os.path.join(OUTPUT_DIR, f"box_particle_counts_member_{m}.png"),
                save=args.save,
                spawn_lon=spawn_lon, spawn_lat=spawn_lat
            )

    ds.close()


if __name__ == "__main__":
    main()
