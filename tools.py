# ===============================================================
# IMPORTING ENVIRONMENT 
# ===============================================================

import os
import sys
import numpy as np
import xarray as xr
import scipy.interpolate as itp
import parcels
import matplotlib.pyplot as plt
import matplotlib.collections as clc
import matplotlib.colors as clr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as mg
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Earth's radius in km
R_EARTH = 6371.0


# ===============================================================
# ===============================================================

def simple_partition_function(coords, mpi_size=1):
    """
    A very simple partition function
    that assigns particles to processors.
    In replacement to scipy/dask methods
    """
    return np.linspace(0, mpi_size, coords.shape[0], endpoint=False, dtype=np.int32)


# ===============================================================
# ===============================================================

def ArgoVerticalMovement(particle, fieldset, time):
    """
    A custom kernel for OceanParcels that mimics the
    behavior of an ARGO float. Unnecessary when using
    VirtualFleet.
    """
    driftdepth = 1000  # maximum depth in m
    maxdepth = 2000  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 10 * 86400  # total time of cycle in seconds
    drifttime = 9 * 86400  # time of deep drift in seconds

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle_ddepth += vertical_speed * particle.dt
        if particle.depth + particle_ddepth >= driftdepth:
            particle_ddepth = driftdepth - particle.depth
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.drift_age += particle.dt
        if particle.drift_age >= drifttime:
            particle.drift_age = 0  # reset drift_age for next cycle
            particle.cycle_phase = 2

    elif particle.cycle_phase == 2:
        # Phase 2: Sinking further to maxdepth
        particle_ddepth += vertical_speed * particle.dt
        if particle.depth + particle_ddepth >= maxdepth:
            particle_ddepth = maxdepth - particle.depth
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle_ddepth -= vertical_speed * particle.dt
        # particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth + particle_ddepth <= fieldset.mindepth:
            particle_ddepth = fieldset.mindepth - particle.depth
            # particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        if particle.cycle_age > cycletime:
            particle.cycle_phase = 0
            particle.cycle_age = 0

    if particle.state == StatusCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age


# ===============================================================
# ===============================================================

def DeleteParticle(particle, fieldset, time):
    
    if particle.state >= 50:
        particle.delete()


# ===============================================================
# ===============================================================

def sample_locations(lat0, lon0, sigma_km, n_samples=1):
    
    """
    Sample random geographic coordinates around a reference point
    with a Gaussian distribution on the Earth's surface.

    Parameters:
    lat0, lon0 : float
        Reference location in degrees.
    sigma_km : float
        Standard deviation of the Gaussian in kilometers.
    n_samples : int
        Number of samples to generate.

    Returns:
    samples : list of (lat, lon) tuples in degrees.
    """
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    # Sample distances with Rayleigh distribution
    distances = np.random.rayleigh(scale=sigma_km, size=n_samples)

    # Sample azimuths uniformly
    azimuths = np.random.uniform(0, 2 * np.pi, n_samples)

    samples = []
    for d, az in zip(distances, azimuths):
        # Angular distance (in radians)
        ang_dist = d / R_EARTH

        # Spherical law of cosines / haversine formula
        lat = np.arcsin(np.sin(lat0_rad) * np.cos(ang_dist) +
                        np.cos(lat0_rad) * np.sin(ang_dist) * np.cos(az))

        lon = lon0_rad + np.arctan2(np.sin(az) * np.sin(ang_dist) * np.cos(lat0_rad),
                                    np.cos(ang_dist) - np.sin(lat0_rad) * np.sin(lat))

        # Normalize lon to [-180, 180]
        lon = (lon + np.pi) % (2 * np.pi) - np.pi

        samples.append((np.degrees(lat), np.degrees(lon)))

    return samples


# =================================================================
# =================================================================

def generate_particles(lat_ref, lon_ref, radius_km, num_particles):
    
    """
    Génère des particules selon une distribution gaussienne autour d'un point de référence
    donné en coordonnées sphériques (latitude, longitude).
    
    :param lat_ref: Latitude du point de référence (en degrés)
    :param lon_ref: Longitude du point de référence (en degrés)
    :param radius_km: Rayon de la distribution gaussienne en kilomètres
    :param num_particles: Nombre de particules à générer
    :return: Liste de tuples (latitude, longitude) des particules
    """
    
    particles = []

    # Convertir latitude et longitude en radians pour calculer les déplacements
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Générer les déviations gaussiennes pour les particules
    dev_lat = np.random.normal(0, radius_km / 111.32, num_particles)  # Conversion km -> degrés lat
    dev_lon = np.random.normal(0, radius_km / (111.32 * np.cos(lat_ref_rad)), num_particles)  # Conversion km -> degrés lon

    for i in range(num_particles):
        # Calculer les nouvelles coordonnées en ajoutant les déviations
        new_lat = lat_ref + dev_lat[i]
        new_lon = lon_ref + dev_lon[i]

        # Ajouter les nouvelles coordonnées à la liste des particules
        particles.append((new_lat, new_lon))

    particles = np.array(particles)

    return particles


# =================================================================
# =================================================================

def get_bathy(file, file_bathy_bin, n=0, force_reextract=False):

    """
    Retrieves depth field from CROCO file (member n+1 of an ensemble,
    default : 1st), and saves it to a .zarr output.
    
    Parameters
    ----------
    file : str
        Path to the CROCO output from which the routine porcesses the extraction.
    file_bathy_bin : str
        Path where to save the depth field.
    force_reextract : bool ; default : False
        Set this flag to True if you want to overwrite an existing .zarr file.
    """

    print()
    if not force_reextract and os.path.exists(file_bathy_bin):
        print(f" *** {file_bathy_bin} already exists! Exiting... ***")
        print()
        sys.exit()

    with xr.open_dataset(file[n]) as ds:
        h = ds['h'].values
        lon = ds['lon_rho'].values
        lat = ds['lat_rho'].values
        mask = ds['mask_rho'].values

    bathy_ds = xr.Dataset({"h": (["y", "x"], h),"lon": (["y", "x"], lon),
                           "lat": (["y", "x"], lat),"mask_rho": (["y", "x"], mask),})
    
    bathy_ds.to_zarr(file_bathy_bin, mode="w")

    print(f" *** Bathy correctly extracted from {file[n]} *** ")
    print(f" *** TYPE : {str(type(h))} *** ")
    print(f" *** SHAPE : {str(np.shape(h))} *** ")
    print(f" *** SAVED TO : {file_bathy_bin} *** ")
    print(" *** Done! *** ")
    print()


# =================================================================
# =================================================================

def get_ssh(file, file_ssh_bin, force_reextract=False):

    """
    Retrieves sea surface height field from an ensemble of CROCO files,
    and saves it to .zarr outputs.
    
    Parameters
    ----------
    file : str
        Path to the CROCO output from which the routine porcesses the extraction.
    file_ssh_bin : str
        Path where to save the sea surface height field.
    force_reextract : bool ; default : False
        Set this flag to True if you want to overwrite an existing .zarr file.
    """
    
    for n in range(len(file)) :

        print()
        if not force_reextract and os.path.exists(file_ssh_bin[n]):
            print(f" *** {file_ssh_bin[n]} already exists! Skipping... ***")
            print()
            continue
            # sys.exit()

        with xr.open_dataset(file[n]) as ds:
            ssh = ds['zeta'].values
            lon = ds['lon_rho'].values
            lat = ds['lat_rho'].values
            mask = ds['mask_rho'].values       

        ssh_ds = xr.Dataset({"zeta": (["t", "y", "x"], ssh),"lon": (["y", "x"], lon),
                             "lat": (["y", "x"], lat),"mask_rho": (["y", "x"], mask),})
        
        ssh_ds.to_zarr(file_ssh_bin[n], mode="w")

        print(f" *** SSH correctly extracted from {file[n]} *** ")
        print(f" *** TYPE : {str(type(ssh))} *** ")
        print(f" *** SHAPE : {str(np.shape(ssh))} *** ")
        print(f" *** SAVED TO : {file_ssh_bin[n]} *** ")
        print(" *** Done! *** ")
        print()


# =================================================================
# =================================================================

def regrid_curv2reg(lon, lat, data, lon_res=0.1, lat_res=0.1,
                        method="linear", max_points=2000):
    """
    Regrids CROCO curvilinear data (lon, lat, data) onto a regular grid
    suitable for contourf or pcolormesh visualization.

    Parameters
    ----------
    lon, lat : 2D arrays
        Curvilinear longitude and latitude (same shape as data).
    data : 2D array
        Field to interpolate (e.g., bathymetry).
    lon_res, lat_res : float
        Target grid resolution (in degrees).
    method : str
        Interpolation method for scipy.interpolate.griddata
        ('linear', 'nearest', or 'cubic').
    max_points : int
        Maximum grid points allowed in each direction (to prevent memory overflow).

    Returns
    -------
    lon_reg, lat_reg, data_reg : 2D arrays
        Regular-grid longitude, latitude, and interpolated field.
    """

    # Step 1 — Clean invalid coordinates
    mask_invalid = (np.abs(lon) > 400) | (np.abs(lat) > 90) | np.isnan(lon) | np.isnan(lat)
    if np.any(mask_invalid):
        print(f"Found {np.sum(mask_invalid)} invalid (lon, lat) points — masking them.")
        lon = np.where(mask_invalid, np.nan, lon)
        lat = np.where(mask_invalid, np.nan, lat)
        data = np.where(mask_invalid, np.nan, data)

    # Remove NaNs for interpolation
    valid_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(data)
    lon_valid = lon[valid_mask]
    lat_valid = lat[valid_mask]
    data_valid = data[valid_mask]

    # Step 2 — Define regular grid bounds
    lon_min, lon_max = np.nanmin(lon_valid), np.nanmax(lon_valid)
    lat_min, lat_max = np.nanmin(lat_valid), np.nanmax(lat_valid)

    lon_points = int((lon_max - lon_min) / lon_res)
    lat_points = int((lat_max - lat_min) / lat_res)

    # Automatically coarsen if grid would be too large
    if lon_points > max_points or lat_points > max_points:
        scale = max(lon_points / max_points, lat_points / max_points)
        lon_res *= scale
        lat_res *= scale
        print(f"Grid too large — auto-adjusted resolution to lon_res={lon_res:.3f}, lat_res={lat_res:.3f}")

    # Create regular grid
    lon_reg_1d = np.arange(lon_min, lon_max + lon_res, lon_res)
    lat_reg_1d = np.arange(lat_min, lat_max + lat_res, lat_res)
    lon_reg, lat_reg = np.meshgrid(lon_reg_1d, lat_reg_1d)

    print(f"Creating regular grid: {len(lat_reg_1d)} × {len(lon_reg_1d)} = {len(lat_reg_1d) * len(lon_reg_1d):,} cells")

    # Step 3 — Interpolate onto the regular grid
    points = np.column_stack((lon_valid, lat_valid))
    data_reg = itp.griddata(points, data_valid, (lon_reg, lat_reg), method=method)

    print("Regridding complete.")
    return lon_reg, lat_reg, data_reg


# =================================================================
# =================================================================

def plot_bathy(lon_reg, lat_reg, h_reg, show_isobaths=True, savepath=None):

    """
    NOTE : As this routine uses contour and contourf mapping, all mappables
    (lon_reg, lat_reg and h_reg) MUST follow a REGULAR grid (arrays of same
    dimension with evenly-spaced values). If you want to visualize ROMS or
    NEMO-like outputs (that is a curvilinear grid), use regrid_curv2reg()
    function before plotting.
    """

    plt.figure(figsize=(10, 8))
    plt.contourf(lon_reg, lat_reg, h_reg, levels=100, cmap="Blues") # cmap = "turbo" or "Blues" or "ocean"
    plt.colorbar(label="Depth [m]")
    if show_isobaths:
        isobath_levels = [100,200,500,1000,1500,2000,2500,4000,4500,5000,5500,6000] # isobaths [m]
        isobaths = plt.contour(lon_reg, lat_reg, h_reg, levels=isobath_levels, colors="black", linewidths=0.9)
        plt.clabel(isobaths, inline=True, fmt="%d m", fontsize=8) # Add labels on isobaths
    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")
    plt.title("SWIOSE Bathymetry at Rho points with regular regridding")
    plt.tight_layout()

    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def map_bathy(lon_reg, lat_reg, h_reg, fill=True, show_isobaths=True, savepath=None):

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(11,9), subplot_kw={'projection': proj})
    
    if fill:
        cs = ax.contourf(lon_reg, lat_reg, h_reg,levels=60,cmap="Blues",transform=ccrs.PlateCarree())
        # cmap = "turbo" or "Blues" or "ocean"
        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label("Depth [m]")
    
    if show_isobaths:
        isobath_levels = [100,200,500,1000,1500,2000,2500,4000,4500,5000,5500,6000] # isobaths [m]
        isobaths = plt.contour(lon_reg, lat_reg, h_reg, levels=isobath_levels, colors="black",
                                linewidths=0.9, transform=ccrs.PlateCarree())
        plt.clabel(isobaths, inline=True, fmt="%d m", fontsize=8) # Add labels on isobaths
    
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False      # disable top labels
    gl.right_labels = False    # disable right labels
    gl.xformatter = mg.LONGITUDE_FORMATTER
    gl.yformatter = mg.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}

    #cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
    #cbar.set_label("Bathymetry (m)")
    ax.set_title("SWIOSE Bathymetry at Rho points with regular regridding", fontsize=15)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    plt.tight_layout()
    
    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def map_ssh(lon_reg, lat_reg, ssh_reg, index=0, fill=True, show_isos=False, savepath=None):

    ssh_reg = ssh_reg[index] # which list element to display, first by default
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(11,9), subplot_kw={'projection': proj})

    if fill:
        cs = ax.contourf(lon_reg, lat_reg, ssh_reg, levels=60, cmap="turbo", transform=ccrs.PlateCarree())
        # cmap = "turbo" or "Blues" or "ocean"
        cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label("Sea surface Height [m]")

    if show_isos:
        isobath_levels = [-0.01,0,0.01,0.02,0.05,0.1,0.15] # Free surface contours [m]
        isobaths = plt.contour(lon_reg, lat_reg, ssh_reg, levels=isobath_levels, colors="black",
                                linewidths=0.9, transform=ccrs.PlateCarree())
        plt.clabel(isobaths, inline=True, fmt="%d m", fontsize=8) # Add labels on isobaths

    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False      # disable top labels
    gl.right_labels = False    # disable right labels
    gl.xformatter = mg.LONGITUDE_FORMATTER
    gl.yformatter = mg.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}

    #cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
    #cbar.set_label("Bathymetry (m)")
    ax.set_title(f"SWIOSE sea surface height (Member {index+1}", fontsize=15)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    plt.tight_layout()

    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath[index]}... *** ")
        plt.savefig(savepath[index], dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def map_ssh_multi_OLD(lon_reg, lat_reg, mappables, fill=True, show_isos=False, savepath=None):
    
    """
    Display six scalar mappables (e.g., SSH fields) in a 3x2 grid of subplots.

    Args:
        lon_reg (2D array): Longitudes grid.
        lat_reg (2D array): Latitudes grid.
        mappables (list): List of 6 arrays (each scalar field to plot).
        fill (bool): Use contourf if True.
        show_isos (bool): Optionally overlay contour lines.
        savepath (str, optional): Path to save the figure.
    """

    if len(mappables) != 6:
        raise ValueError("Expected exactly 6 mappables in the 'mappables' list.")

    cs_list = []
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={'projection': proj})
    axes = axes.flatten()

    for i, (ax, ssh_reg) in enumerate(zip(axes, mappables)):
        if fill:
            cs = ax.contourf(
                lon_reg, lat_reg, ssh_reg, levels=60, cmap="turbo", transform=proj
            )
            # cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
            # cbar.set_label("Sea surface Height [m]")
            cs_list.append(cs)

        if show_isos:
            isobath_levels = [-0.01, 0, 0.01, 0.02, 0.05, 0.1, 0.15]
            isobaths = ax.contour(
                lon_reg, lat_reg, ssh_reg, levels=isobath_levels,
                colors="black", linewidths=0.7, transform=proj
            )
            ax.clabel(isobaths, inline=True, fmt="%d m", fontsize=7)

        ax.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = mg.LONGITUDE_FORMATTER
        gl.yformatter = mg.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        ax.set_title(f"SSH Map #{i + 1}", fontsize=12)

    if fill:
        cbar = fig.colorbar(cs_list[0], ax=axes, orientation='vertical', shrink=0.85, pad=0.02)
        cbar.set_label("Sea surface Height [m]")

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    if savepath:
        print(f"*** Saving 6-panel figure to {savepath} ***")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        plt.show()


# =================================================================
# =================================================================

def map_ssh_multi(lon_reg, lat_reg, mappables, fill=True,
                  show_isos=False, savepath=None, cmap="turbo"):
    """
    Display six scalar mappables in a 3x2 grid with shared color scale and one colorbar.
    """

    if len(mappables) != 6:
        raise ValueError("Expected exactly 6 mappables in 'mappables' list.")

    vmin = min(np.nanmin(m) for m in mappables)
    vmax = max(np.nanmax(m) for m in mappables)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(15,9), subplot_kw={'projection': proj})
    axes = axes.flatten()

    cs_list = []
    for i, (ax, ssh_reg) in enumerate(zip(axes, mappables)):
        if fill:
            cs = ax.contourf(
                lon_reg, lat_reg, ssh_reg, levels=60, cmap=cmap,
                transform=proj, vmin=vmin, vmax=vmax
            )
            cs_list.append(cs)

        if show_isos:
            isobath_levels = [-0.01, 0, 0.01, 0.02, 0.05, 0.1, 0.15]
            isobaths = ax.contour(
                lon_reg, lat_reg, ssh_reg, levels=isobath_levels,
                colors="black", linewidths=0.7, transform=proj
            )
            ax.clabel(isobaths, inline=True, fmt="%d m", fontsize=7)

        ax.coastlines(resolution='10m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = mg.LONGITUDE_FORMATTER
        gl.yformatter = mg.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        ax.set_title(f"Sea Surface Height (Member #{i + 1})", fontsize=12)

    # --- Shared colorbar placed outside ---
    if fill:
        fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.15)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(cs_list[0], cax=cbar_ax)
        cbar.set_label("Sea surface Height [m]")

    else:
        plt.tight_layout()

    if savepath:
        print(f"*** Saving 6-panel figure to {savepath} ***")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        plt.show()


# =================================================================
# =================================================================

def spaghetti(x, y, x0, y0, npart, ensemble, savepath=None):

    #plt.close('all')
    plt.figure(figsize=(10, 8))
    for idx in range(len(x)):
        plt.plot(x[idx].flatten(), y[idx].flatten(), linestyle='solid', linewidth=1, color='grey')
    plt.plot(x0, y0, 'ko', markersize=5, label="Initial coordinates")
    plt.xlabel("Zonal drift [°Longitude]", fontsize=12)
    plt.ylabel("Meridional drift [°Latitude]", fontsize=12)
    plt.title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.legend(loc='best')
    plt.tight_layout()

    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def colorplot_time(x, y, x0, y0, npart, ensemble, savepath=None):

    #plt.close('all')
    plt.figure(figsize=(10, 6))

    ax = plt.gca()
    lengths = [len(x[idx].flatten()) for idx in range(len(x))]
    max_len = max(lengths)
    norm = clr.Normalize(vmin=0, vmax=max_len)
    cmap = plt.get_cmap('viridis')

    for idx in range(len(x)):

        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #norm = clr.Normalize(vmin=0, vmax=len(x_flat))
        lc = clc.LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
        lc.set_array(np.arange(len(x_flat)))
        line = ax.add_collection(lc)
        # plt.plot(x[idx].flatten(), y[idx].flatten(), linestyle='solid', linewidth=1, color='grey')

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Progression over time (days)')

    plt.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates")
    plt.xlabel("Zonal drift [°Longitude]", fontsize=12)
    plt.ylabel("Meridional drift [°Latitude]", fontsize=12)
    plt.title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.legend(loc='best')
    plt.tight_layout()

    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# ===================================================================
# ===================================================================

def colormap_time(x, y, x0, y0, npart, ensemble, savepath=None, background=True):
    
    """
    Plots particle trajectories over time,with
    cartopy coastlines for geographic context.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    lengths = [len(x[idx].flatten()) for idx in range(len(x))]
    max_len = max(lengths)
    norm = clr.Normalize(vmin=0, vmax=max_len)
    cmap = plt.get_cmap('viridis')

    for idx in range(len(x)):
        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = clc.LineCollection(segments, cmap=cmap, norm=norm, linewidth=1, transform=ccrs.PlateCarree())
        lc.set_array(np.arange(len(x_flat)))
        ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Progression over time (days)')

    ax.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates", transform=ccrs.PlateCarree())
    
    if background:
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=0)
    
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    ax.set_title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s)", fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()

    if savepath:
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# ===================================================================
# ===================================================================

def colormap_time_bathy(x, y, x0, y0, npart, ensemble, dpt, show_isobaths=True,
                        lon_reg=None, lat_reg=None, h_reg=None, savepath=None):
    """
    Plots particle trajectories over time with optional bathymetric overlay.
    """

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    background = True if dpt==-10 else False
    lengths = [len(x[idx].flatten()) for idx in range(len(x))]
    max_len = max(lengths)
    norm = clr.Normalize(vmin=0, vmax=max_len)
    cmap = plt.get_cmap('viridis')

    for idx in range(len(x)):
        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = clc.LineCollection(segments, cmap=cmap, norm=norm, linewidth=1, transform=ccrs.PlateCarree())
        lc.set_array(np.arange(len(x_flat)))
        ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Progression over time (days)')

    ax.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates", transform=ccrs.PlateCarree())

    if background:
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=0)

    # === Add isobaths if requested ===
    if show_isobaths and lon_reg is not None and lat_reg is not None and h_reg is not None:
        if dpt == -10 :
            isobath_levels = [200,500,1000,2000,4000,5000,6000]
        elif dpt == -1000 :
            isobath_levels = [1000,2000,4000,5000,6000]
        iso0 = ax.contour(lon_reg, lat_reg, h_reg, levels=[isobath_levels[0]], colors='black',
                          linewidths=1.3, transform=ccrs.PlateCarree())
        isobaths = ax.contour(lon_reg, lat_reg, h_reg, levels=isobath_levels, colors='grey',
                              linewidths=0.8, transform=ccrs.PlateCarree())
        ax.clabel(iso0, inline=True, fmt="%d m", fontsize=8)
        ax.clabel(isobaths, inline=True, fmt="%d m", fontsize=8)
    
    if background:
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=0)

    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    ax.set_title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s) (z = {abs(dpt)} m)", fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()

    if savepath:
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def colorplot_traj(x, y, x0, y0, npart, ensemble, savepath=None):

    #plt.close('all')
    plt.figure(figsize=(10, 6))

    ax = plt.gca()
    n_traj = len(x)
    norm = clr.Normalize(vmin=0, vmax=n_traj-1)
    cmap = plt.get_cmap('viridis')
    handles = []

    for idx in range(n_traj):

        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        color = cmap(norm(idx))
        lc = clc.LineCollection(segments, colors=[color], linewidth=1, label=f"{idx+1}")
        lc.set_array(np.arange(len(x_flat)))
        line = ax.add_collection(lc)
        handles.append(lc)
        # plt.plot(x[idx].flatten(), y[idx].flatten(), linestyle='solid', linewidth=1, color='grey')

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    #plt.colorbar(sm, ax=ax, label='Ensemble member')

    plt.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates")
    plt.xlabel("Zonal drift [°Longitude]", fontsize=12)
    plt.ylabel("Meridional drift [°Latitude]", fontsize=12)
    plt.title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.legend(loc='best')
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Ensemble members",
            fontsize=11, title_fontsize=11, handleheight=1.5, frameon=False, ncol=2)
    plt.tight_layout()#rect=[0, 0, 0.82, 1]) # leave space for legend

    if savepath:  # only save if user provided a path
        print(f" *** Saving figure to {savepath}... *** ")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def colormap_traj(x, y, x0, y0, npart, ensemble, savepath=None, background=True):

    """
    Plot multiple 2D trajectories (longitude, latitude) with optional Cartopy map features.

    Parameters
    ----------
    x, y : list of np.ndarray
        Lists containing arrays of longitudes and latitudes for each trajectory.
    x0, y0 : float or array-like
        Initial coordinates to highlight on the plot.
    npart : int
        Number of particles (for labeling purposes).
    ensemble : list
        Ensemble members (for labeling).
    savepath : str, optional
        If provided, saves the figure instead of displaying.
    map_features : bool, optional
        If True, overlay trajectories on a Cartopy map background.
    """
    
    plt.figure(figsize=(11,8))

    if background:
        ax = plt.axes(projection=ccrs.PlateCarree())
        # --- add cartopy features ---
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        #ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.LAKES, facecolor='lightblue')
        #ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
        # --- Labeled gridlines ---
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = mg.LONGITUDE_FORMATTER
        gl.yformatter = mg.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        transform = ccrs.PlateCarree()
    else:
        ax = plt.gca()
        transform = None

    n_traj = len(x)
    norm = clr.Normalize(vmin=0, vmax=n_traj - 1)
    cmap = plt.get_cmap('viridis')
    handles = []

    for idx in range(n_traj):
        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        color = cmap(norm(idx))
        lc = clc.LineCollection(
            segments,
            colors=[color],
            linewidth=1,
            label=f"{idx+1}",
            transform=transform)
        lc.set_array(np.arange(len(x_flat)))
        ax.add_collection(lc)
        handles.append(lc)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ax.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates", transform=transform)

    ax.set_xlabel("Longitude [°]", fontsize=12)
    ax.set_ylabel("Latitude [°]", fontsize=12)
    ax.set_title(f"Lagrangian drift of {npart} particle(s) for {len(ensemble)} member(s)", fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title="Ensemble members",
        fontsize=11,
        title_fontsize=11,
        handleheight=1.5,
        frameon=False,
        ncol=2)

    plt.tight_layout()

    if savepath:
        print(f" *** Saving figure to {savepath}... ***")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()


# =================================================================
# =================================================================

def colormap_traj_dt(x, y, x0, y0, npart, steps, savepath=None, background=True):

    """
    Plot multiple 2D trajectories (longitude, latitude) with optional Cartopy map features.

    Parameters
    ----------
    x, y : list of np.ndarray
        Lists containing arrays of longitudes and latitudes for each trajectory.
    x0, y0 : float or array-like
        Initial coordinates to highlight on the plot.
    npart : int
        Number of particles (for labeling purposes).
    ensemble : list
        Ensemble members (for labeling).
    savepath : str, optional
        If provided, saves the figure instead of displaying.
    map_features : bool, optional
        If True, overlay trajectories on a Cartopy map background.
    """

    plt.figure(figsize=(13,8))

    if background:
        ax = plt.axes(projection=ccrs.PlateCarree())
        # --- add cartopy features ---
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        #ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.LAKES, facecolor='lightblue')
        #ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
        # --- Labeled gridlines ---
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = mg.LONGITUDE_FORMATTER
        gl.yformatter = mg.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        transform = ccrs.PlateCarree()
    else:
        ax = plt.gca()
        transform = None

    n_traj = len(x)
    norm = clr.Normalize(vmin=0, vmax=n_traj - 1)
    cmap = plt.get_cmap('viridis')
    handles = []

    for idx in range(n_traj):
        x_flat, y_flat = x[idx].flatten(), y[idx].flatten()
        points = np.array([x_flat, y_flat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        color = cmap(norm(idx))
        lc = clc.LineCollection(
            segments,
            colors=[color],
            linewidth=1,
            label=f"dt = {steps[idx]}s",
            transform=transform)
        lc.set_array(np.arange(len(x_flat)))
        ax.add_collection(lc)
        handles.append(lc)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ax.plot(x0, y0, 'ro', markersize=7, label="Initial coordinates", transform=transform)

    ax.set_xlabel("Longitude [°]", fontsize=12)
    ax.set_ylabel("Latitude [°]", fontsize=12)
    ax.set_title(f"Lagrangian drift of {npart} particle(s) for {len(steps)} different computing timesteps", fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title="Timestep",
        fontsize=14,
        title_fontsize=14,
        handleheight=1.5,
        frameon=False,
        ncol=1)

    plt.tight_layout()

    if savepath:
        print(f" *** Saving figure to {savepath}... ***")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
    else:
        print(" *** Displaying figure on terminal... *** ")
        plt.show()

