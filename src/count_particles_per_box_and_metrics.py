#!/usr/bin/env python3
# ================================================================
# COUNT PARTICLES PER SUBDOMAIN BOX
# ----------------------------------------------------------------
# For each ensemble member's .zarr trajectory store produced by
# runpar_ini_opt_2Dgauss.py, this script:
#   1) divides the simulation domain into square boxes of a given
#      side length (default 50 km),
#   2) for each box, counts the number of PARTICLES that make at
#      least one pass through that box at any timestep,
#   3) reports the count broken down (optionally) by:
#        - ensemble member
#        - approximate spawning point (the release point each
#          particle's Gaussian cloud was drawn from)
#        - time of passage (binned into windows of TIME_BIN_DAYS)
#
# IMPORTANT ASYMMETRY BETWEEN BREAKDOWNS
# ----------------------------------------------------------------
# - Summing counts across MEMBERS is always valid (members are
#   independent particle sets).
# - Summing counts across SPAWN POINTS is always valid (each
#   particle belongs to exactly one spawn point, so spawn points
#   partition the particles).
# - Summing counts across TIME BINS is *not* valid: a particle
#   can revisit the same box in several time windows, so summing
#   per-window "at least one pass" counts would double-count it.
#   For this reason, the whole-run "how many particles ever passed
#   through this box" answer is always computed separately (see
#   n_particles_total / n_particles_per_member below), and the
#   time-resolved breakdown is reported independently in the long
#   table rather than being derivable from it by summation.
#
# OUTPUTS
# ----------------------------------------------------------------
#   - box_particle_counts.nc     : dense grid, whole-run counts
#                                   (n_particles_total, n_particles_per_member)
#   - box_particle_counts.csv    : flat version of the above, for quick plotting
#   - box_passages_long.csv      : long-format table with the requested
#                                   breakdown (member / spawn point / time bin),
#                                   one row per non-empty combination. Includes
#                                   exact integer box indices (ilat, ilon) so
#                                   downstream scripts (e.g. the plotting
#                                   script) can rebuild a grid without relying
#                                   on floating-point coordinate matching.
#   - connectivity_matrix.csv    : site x site matrix, raw particle counts
#                                   AND connection probability (see below)
#   - connectivity_metrics.csv   : per-site in-degree and betweenness centrality
#
# SITE-TO-SITE CONNECTIVITY AND GRAPH METRICS (new)
# ----------------------------------------------------------------
# In addition to the subdomain box analysis above, this script builds a
# directed connectivity graph between the release sites themselves:
# for every pair of sites (i, j), it counts how many particles released
# at site i come within SITE_RADIUS_KM of site j at any point in their
# trajectory (summed over the whole ensemble, since summing across
# members is always valid). This is turned into:
#   - connectivity_matrix[i, j]     : raw particle counts (includes the
#                                      diagonal, i.e. self-recruitment)
#   - connectivity_prob[i, j]       : counts[i, j] / (PARTICLES_PER_POINT
#                                      * n_members), i.e. the fraction of
#                                      particles released at i that reach j
#   - in_degree[j]                  : sum over i != j of connectivity_prob,
#                                      i.e. how much incoming connectivity
#                                      a site receives from OTHER sites
#                                      (self-recruitment excluded)
#   - betweenness[j]                : betweenness centrality of site j in
#                                      the directed graph built from
#                                      connectivity_prob (self-loops
#                                      excluded), which identifies
#                                      "stepping stone" sites that lie on
#                                      many likely connection paths.
#                                      Edge distance is -log(p): a
#                                      principled way to turn per-step
#                                      probabilities into additive path
#                                      "costs" (path cost sums to
#                                      -log(product of probabilities)
#                                      along the route).
#
# NEW DEPENDENCIES: this part requires scipy (cKDTree, for fast
# radius search) and networkx (for betweenness centrality). Neither
# was needed by the rest of the script before.
#
# COMMAND-LINE OPTIONS
# ----------------------------------------------------------------
#   --mesh SIZE_KM     override BOX_SIZE_KM below, e.g. --mesh 25
#   --radius RADIUS_KM override SITE_RADIUS_KM below, e.g. --radius 15
# ================================================================

import os
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI
from scipy.spatial import cKDTree

try:
    import networkx as nx
except ImportError:
    nx = None


# ================================================================
# MPI SETUP
# ================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ================================================================
# USER PARAMETERS
# ================================================================

PATH_ABS = 'your_path_to_enslag_connectivities'
TRAJ_DIR = os.path.join(PATH_ABS, "output/ini")
TRAJ_PATTERN = "traj_member_*.zarr"
SPOTS_FILE = os.path.join(PATH_ABS, "input/release_spots.txt")

BOX_SIZE_KM = 50.0   # side length of each square subdomain box

# Domain bounds in degrees. Set to None to auto-detect from the
# trajectory stores (a single lazy min/max pass, cheap).
LON_MIN, LON_MAX = None, None
LAT_MIN, LAT_MAX = None, None

# ---- breakdown options -----------------------------------------
# Must match the values used in runpar_ini_opt_2Dgauss.py so that
# particle index -> spawn point mapping is correct.
PARTICLES_PER_POINT = 4000

GROUP_BY_SPAWN = True     # break passages down by approximate spawning point
GROUP_BY_TIME = True      # break passages down by time-of-passage window
TIME_BIN_DAYS = 5.0       # width of each time-of-passage window, in days
                          # (only used if GROUP_BY_TIME is True)

# ---- site-to-site connectivity / graph metrics -----------------
COMPUTE_CONNECTIVITY = True
SITE_RADIUS_KM = 25.0     # a particle "arrives" at a site if it comes within
                          # this radius of the site's coordinates at any point

OUTPUT_NC = os.path.join(PATH_ABS, "output/box_particle_counts.nc")
OUTPUT_CSV = os.path.join(PATH_ABS, "output/box_particle_counts.csv")
OUTPUT_LONG_CSV = os.path.join(PATH_ABS, "output/box_passages_long.csv")
OUTPUT_CONNECTIVITY_CSV = os.path.join(PATH_ABS, "output/connectivity_matrix.csv")
OUTPUT_METRICS_CSV = os.path.join(PATH_ABS, "output/connectivity_metrics.csv")


# ================================================================
# LOAD RELEASE POINTS (spawn point coordinates, for reference)
# ================================================================

def load_release_points():
    data = np.loadtxt(SPOTS_FILE, comments='#')
    lon = data[:, 0]
    lat = data[:, 1]
    return lon, lat


# ================================================================
# CLI ARGUMENTS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count particles passing through each subdomain box."
    )
    parser.add_argument(
        "--mesh", type=float, default=None,
        help=f"Box side length in km (default: {BOX_SIZE_KM})."
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help=f"Site connectivity radius in km (default: {SITE_RADIUS_KM})."
    )
    return parser.parse_args()


# ================================================================
# GRID CONSTRUCTION
# ================================================================

KM_PER_DEG_LAT = 111.32


def km_to_deg_lat(km):
    return km / KM_PER_DEG_LAT


def km_to_deg_lon(km, lat_ref):
    return km / (KM_PER_DEG_LAT * np.cos(np.radians(lat_ref)))


def build_grid(lon_min, lon_max, lat_min, lat_max, box_km):
    """Build box edges (lon_edges, lat_edges) such that each box is
    approximately box_km x box_km, using the domain's mean latitude
    as the reference for the longitude -> degrees conversion."""

    lat_ref = 0.5 * (lat_min + lat_max)
    dlat = km_to_deg_lat(box_km)
    dlon = km_to_deg_lon(box_km, lat_ref)

    lat_edges = np.arange(lat_min, lat_max + dlat, dlat)
    lon_edges = np.arange(lon_min, lon_max + dlon, dlon)

    return lon_edges, lat_edges


def flat_box_index(lon, lat, lon_edges, lat_edges):
    """Map (lon, lat) arrays (any shape) to a flat box index.
    Returns -1 for points outside the grid or NaN (e.g. deleted
    particles)."""

    nlon = len(lon_edges) - 1
    nlat = len(lat_edges) - 1

    ilon = np.digitize(lon, lon_edges) - 1
    ilat = np.digitize(lat, lat_edges) - 1

    valid = (
        (ilon >= 0) & (ilon < nlon) &
        (ilat >= 0) & (ilat < nlat) &
        np.isfinite(lon) & np.isfinite(lat)
    )

    flat = np.full(lon.shape, -1, dtype=np.int64)
    flat[valid] = ilat[valid] * nlon + ilon[valid]

    return flat, nlon, nlat


def to_local_km(lon, lat, lon_ref, lat_ref):
    """Project (lon, lat) to a local equirectangular (x_km, y_km) plane
    centred on (lon_ref, lat_ref), using the same local degree->km
    conversion as build_grid. Good enough for radius searches over a
    regional domain; not a substitute for a real geodesic projection
    over very large domains."""

    x_km = (lon - lon_ref) * KM_PER_DEG_LAT * np.cos(np.radians(lat_ref))
    y_km = (lat - lat_ref) * KM_PER_DEG_LAT
    return x_km, y_km


# ================================================================
# DOMAIN BOUNDS AUTO-DETECTION
# ================================================================

def get_domain_bounds(files):
    """Single lazy pass (dask-backed min/max) over all trajectory
    stores to get the overall lon/lat extent."""

    lon_min, lon_max = np.inf, -np.inf
    lat_min, lat_max = np.inf, -np.inf

    for f in files:
        ds = xr.open_zarr(f)
        lon_min = min(lon_min, float(ds['lon'].min().compute()))
        lon_max = max(lon_max, float(ds['lon'].max().compute()))
        lat_min = min(lat_min, float(ds['lat'].min().compute()))
        lat_max = max(lat_max, float(ds['lat'].max().compute()))
        ds.close()

    return lon_min, lon_max, lat_min, lat_max


# ================================================================
# PER-MEMBER PROCESSING
# ================================================================

def process_member(file, lon_edges, lat_edges, member_name,
                    group_by_spawn, particles_per_point,
                    group_by_time, time_bin_days,
                    compute_connectivity=False, site_x=None, site_y=None,
                    site_radius_km=None, lon_ref=None, lat_ref=None):
    """Process one member's zarr store.

    Returns
    -------
    dense_counts : 1D array (nlon*nlat)
        Number of distinct particles visiting each box at least
        once over the WHOLE run (no spawn / time breakdown). This
        is the safely-summable whole-run answer.
    long_df : DataFrame or None
        Long-format table with the requested breakdown, columns
        among ['member', 'spawn_id', 'time_bin', 'cell',
        'n_particles']. None if neither breakdown was requested.
    connectivity : 2D array (n_sites, n_sites) or None
        connectivity[i, j] = number of distinct particles released at
        site i (this member only) that came within site_radius_km of
        site j at any point. None if compute_connectivity is False.
    """

    nlon = len(lon_edges) - 1
    nlat = len(lat_edges) - 1

    ds = xr.open_zarr(file)
    lon = ds['lon'].values     # shape (traj, obs)
    lat = ds['lat'].values
    time = ds['time'].values if group_by_time else None
    ds.close()

    flat, _, _ = flat_box_index(lon, lat, lon_edges, lat_edges)
    ntraj, nobs = flat.shape

    traj_idx = np.repeat(np.arange(ntraj), nobs)
    flat_r = flat.ravel()
    mask = flat_r >= 0

    # ---- whole-run dense grid (no spawn / no time breakdown) ----
    df_base = pd.DataFrame({'traj': traj_idx[mask], 'cell': flat_r[mask]}).drop_duplicates()
    dense_counts = np.zeros(nlon * nlat, dtype=np.int64)
    vc = df_base['cell'].value_counts()
    dense_counts[vc.index.to_numpy()] = vc.to_numpy()

    # ---- optional long-format breakdown ----
    long_df = None
    if group_by_spawn or group_by_time:

        data = {'traj': traj_idx[mask], 'cell': flat_r[mask]}
        group_cols = ['cell']

        if group_by_spawn:
            data['spawn_id'] = (traj_idx // particles_per_point)[mask]
            group_cols.append('spawn_id')

        if group_by_time:
            t = time.astype('datetime64[s]').astype('float64')  # seconds
            t0 = np.nanmin(t)
            elapsed_days = (t - t0) / 86400.0
            bin_idx = np.floor(elapsed_days / time_bin_days).astype(np.int64)
            data['time_bin'] = bin_idx.ravel()[mask]
            group_cols.append('time_bin')

        df = pd.DataFrame(data).drop_duplicates(subset=['traj'] + group_cols)
        long_df = df.groupby(group_cols).size().reset_index(name='n_particles')
        long_df['member'] = member_name

    # ---- optional site-to-site connectivity ----
    connectivity = None
    if compute_connectivity:
        n_sites = len(site_x)

        x, y = to_local_km(lon, lat, lon_ref, lat_ref)
        x_flat = x.ravel()
        y_flat = y.ravel()
        pt_mask = np.isfinite(x_flat) & np.isfinite(y_flat)

        points = np.column_stack([x_flat[pt_mask], y_flat[pt_mask]])
        pt_traj_idx = traj_idx[pt_mask]

        connectivity = np.zeros((n_sites, n_sites), dtype=np.int64)

        if len(points) > 0:
            tree = cKDTree(points)
            for j in range(n_sites):
                idxs = tree.query_ball_point([site_x[j], site_y[j]], r=site_radius_km)
                if len(idxs) == 0:
                    continue
                visiting_traj = np.unique(pt_traj_idx[idxs])
                spawn_ids = visiting_traj // particles_per_point
                connectivity[:, j] += np.bincount(spawn_ids, minlength=n_sites)

    return dense_counts, long_df, connectivity


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    args = parse_args()
    box_size_km = args.mesh if args.mesh is not None else BOX_SIZE_KM
    site_radius_km = args.radius if args.radius is not None else SITE_RADIUS_KM

    files = sorted(glob.glob(os.path.join(TRAJ_DIR, TRAJ_PATTERN)))

    if len(files) == 0:
        if rank == 0:
            print(f"No trajectory files found in {TRAJ_DIR} "
                  f"matching {TRAJ_PATTERN}", flush=True)
        return

    if rank == 0:
        print(f"\nFound {len(files)} member trajectory stores", flush=True)

    # ------------------------------------------------------------
    # domain bounds (rank 0 computes, then broadcasts)
    # ------------------------------------------------------------
    if rank == 0:
        if None in (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX):
            print("Auto-detecting domain bounds...", flush=True)
            lon_min, lon_max, lat_min, lat_max = get_domain_bounds(files)
        else:
            lon_min, lon_max = LON_MIN, LON_MAX
            lat_min, lat_max = LAT_MIN, LAT_MAX
        bounds = (lon_min, lon_max, lat_min, lat_max)
    else:
        bounds = None

    bounds = comm.bcast(bounds, root=0)
    lon_min, lon_max, lat_min, lat_max = bounds

    lon_edges, lat_edges = build_grid(lon_min, lon_max, lat_min, lat_max, box_size_km)
    nlon = len(lon_edges) - 1
    nlat = len(lat_edges) - 1
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    if rank == 0:
        print(f"Domain bounds:  lon [{lon_min:.3f}, {lon_max:.3f}]  "
              f"lat [{lat_min:.3f}, {lat_max:.3f}]", flush=True)
        print(f"Box grid:       {nlon} x {nlat} boxes "
              f"({box_size_km} km side)", flush=True)
        print(f"Breakdown:      spawn_point={GROUP_BY_SPAWN}, "
              f"time (bin={TIME_BIN_DAYS}d)={GROUP_BY_TIME}", flush=True)
        print(f"Connectivity:   {COMPUTE_CONNECTIVITY} "
              f"(site radius = {site_radius_km} km)", flush=True)

    # ------------------------------------------------------------
    # release sites, projected to a local km plane for the
    # connectivity radius search (same reference point on all ranks)
    # ------------------------------------------------------------
    site_lon, site_lat, site_x, site_y, lon_ref, lat_ref = (None,) * 6
    if COMPUTE_CONNECTIVITY:
        site_lon, site_lat = load_release_points()
        n_sites = len(site_lon)
        lon_ref = 0.5 * (lon_min + lon_max)
        lat_ref = 0.5 * (lat_min + lat_max)
        site_x, site_y = to_local_km(site_lon, site_lat, lon_ref, lat_ref)

        if rank == 0:
            print(f"Release sites:  {n_sites} (from {SPOTS_FILE})", flush=True)

    # ------------------------------------------------------------
    # split members across ranks
    # ------------------------------------------------------------
    local_files = files[rank::size]
    print(f"[rank {rank:02d}] Members assigned: {len(local_files)}", flush=True)

    member_names = []
    member_dense = []
    long_dfs = []
    member_connectivity = []

    for f in local_files:
        member = os.path.basename(f).split('_')[-1].replace('.zarr', '')
        print(f"[rank {rank}] Counting boxes for member {member}", flush=True)

        dense_counts, long_df, connectivity = process_member(
            f, lon_edges, lat_edges, member,
            GROUP_BY_SPAWN, PARTICLES_PER_POINT,
            GROUP_BY_TIME, TIME_BIN_DAYS,
            compute_connectivity=COMPUTE_CONNECTIVITY,
            site_x=site_x, site_y=site_y,
            site_radius_km=site_radius_km,
            lon_ref=lon_ref, lat_ref=lat_ref
        )

        member_names.append(member)
        member_dense.append(dense_counts)
        if long_df is not None:
            long_dfs.append(long_df)
        if connectivity is not None:
            member_connectivity.append(connectivity)

    # ------------------------------------------------------------
    # gather results to rank 0
    # ------------------------------------------------------------
    gathered_names = comm.gather(member_names, root=0)
    gathered_dense = comm.gather(member_dense, root=0)
    gathered_long = comm.gather(long_dfs, root=0)
    gathered_connectivity = comm.gather(member_connectivity, root=0)

    comm.Barrier()

    if rank == 0:

        # ---- dense whole-run grid (unchanged from before) ----
        all_names = [n for sub in gathered_names for n in sub]
        all_dense = [c for sub in gathered_dense for c in sub]

        order = np.argsort(all_names)
        all_names = [all_names[i] for i in order]
        all_dense = np.stack([all_dense[i] for i in order], axis=0)  # (member, nlon*nlat)

        total_counts = all_dense.sum(axis=0)

        per_member_grid = all_dense.reshape(len(all_names), nlat, nlon)
        total_grid = total_counts.reshape(nlat, nlon)

        ds_out = xr.Dataset(
            data_vars={
                "n_particles_total": (("lat_box", "lon_box"), total_grid),
                "n_particles_per_member": (("member", "lat_box", "lon_box"), per_member_grid),
            },
            coords={
                "lon_box": lon_centers,
                "lat_box": lat_centers,
                "member": all_names,
            },
            attrs={
                "description": "Number of particles making at least one pass "
                                "through each subdomain box, over the whole run",
                "box_size_km": box_size_km,
                "n_members": len(all_names),
            },
        )

        os.makedirs(os.path.dirname(OUTPUT_NC), exist_ok=True)
        ds_out.to_netcdf(OUTPUT_NC)
        print(f"\nSaved whole-run grid to {OUTPUT_NC}", flush=True)

        lon_c, lat_c = np.meshgrid(lon_centers, lat_centers)
        df_csv = pd.DataFrame({
            "lon_box": lon_c.ravel(),
            "lat_box": lat_c.ravel(),
            "n_particles_total": total_grid.ravel(),
        })
        df_csv.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved flat table to {OUTPUT_CSV}", flush=True)

        # ---- long-format breakdown table ----
        all_long = [df for sub in gathered_long for df in sub]

        if len(all_long) > 0:
            long_table = pd.concat(all_long, ignore_index=True)

            # cell -> exact integer box indices, kept as the authoritative
            # key for matching against the grid downstream (e.g. in the
            # plotting script) -- avoids floating-point coordinate matching
            long_table['ilat'] = (long_table['cell'] // nlon).astype(np.int64)
            long_table['ilon'] = (long_table['cell'] % nlon).astype(np.int64)
            # human-readable box-center coordinates, for reference only
            long_table['lat_box'] = lat_centers[long_table['ilat'].to_numpy()]
            long_table['lon_box'] = lon_centers[long_table['ilon'].to_numpy()]
            long_table = long_table.drop(columns=['cell'])

            if GROUP_BY_SPAWN:
                spawn_lon, spawn_lat = load_release_points()
                long_table['spawn_lon'] = spawn_lon[long_table['spawn_id'].to_numpy()]
                long_table['spawn_lat'] = spawn_lat[long_table['spawn_id'].to_numpy()]

            if GROUP_BY_TIME:
                long_table['time_bin_start_days'] = long_table['time_bin'] * TIME_BIN_DAYS
                long_table['time_bin_end_days'] = (long_table['time_bin'] + 1) * TIME_BIN_DAYS

            long_table.to_csv(OUTPUT_LONG_CSV, index=False)
            print(f"Saved long-format passages table to {OUTPUT_LONG_CSV} "
                  f"({len(long_table)} rows)", flush=True)
            print("NOTE: rows may be summed across 'member' and/or 'spawn_id' "
                  "safely, but NOT across 'time_bin' (a particle can revisit "
                  "a box in several time windows) -- use n_particles_total / "
                  "n_particles_per_member in the NetCDF file for the whole-run "
                  "count instead.", flush=True)

        # ---- site-to-site connectivity and graph metrics ----
        all_connectivity = [c for sub in gathered_connectivity for c in sub]

        if len(all_connectivity) > 0:
            connectivity_counts = np.sum(all_connectivity, axis=0)  # (n_sites, n_sites)
            n_members = len(all_names)
            n_sites = connectivity_counts.shape[0]

            total_released_per_site = PARTICLES_PER_POINT * n_members
            connectivity_prob = connectivity_counts / total_released_per_site

            # ---- save raw matrix + probability matrix (long format) ----
            src_idx, dst_idx = np.meshgrid(np.arange(n_sites), np.arange(n_sites), indexing='ij')
            df_conn = pd.DataFrame({
                'site_from': src_idx.ravel(),
                'site_to': dst_idx.ravel(),
                'n_particles': connectivity_counts.ravel(),
                'probability': connectivity_prob.ravel(),
            })
            df_conn.to_csv(OUTPUT_CONNECTIVITY_CSV, index=False)
            print(f"\nSaved connectivity matrix to {OUTPUT_CONNECTIVITY_CSV}", flush=True)

            # ---- in-degree (self-recruitment excluded) ----
            off_diag_prob = connectivity_prob.copy()
            np.fill_diagonal(off_diag_prob, 0.0)
            in_degree = off_diag_prob.sum(axis=0)  # sum over sources i, per destination j

            self_recruitment = np.diag(connectivity_prob)

            # ---- betweenness centrality ----
            betweenness = np.full(n_sites, np.nan)
            if nx is None:
                print("WARNING: networkx is not installed -- skipping betweenness "
                      "centrality (in_degree and self-recruitment are still "
                      "computed). Install with 'pip install networkx'.", flush=True)
            else:
                G = nx.DiGraph()
                G.add_nodes_from(range(n_sites))
                for i in range(n_sites):
                    for j in range(n_sites):
                        if i != j and connectivity_prob[i, j] > 0:
                            G.add_edge(i, j, distance=-np.log(connectivity_prob[i, j]))
                betweenness_dict = nx.betweenness_centrality(G, weight='distance', normalized=True)
                betweenness = np.array([betweenness_dict[i] for i in range(n_sites)])

            df_metrics = pd.DataFrame({
                'site_id': np.arange(n_sites),
                'lon': site_lon,
                'lat': site_lat,
                'in_degree': in_degree,
                'self_recruitment': self_recruitment,
                'betweenness': betweenness,
            })
            df_metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
            print(f"Saved per-site connectivity metrics to {OUTPUT_METRICS_CSV}", flush=True)
            print("NOTE: in_degree and betweenness both exclude self-recruitment "
                  "(the diagonal); self_recruitment is reported separately. "
                  "Betweenness uses edge distance = -log(probability).", flush=True)

        print("\nAll done.", flush=True)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
