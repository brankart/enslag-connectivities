# file: src/plot_polygons.py

import os
import errno
import argparse
from pathlib import Path
from typing import List
import numpy as np
import xarray as xr
from rasterio import features
from affine import Affine
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    SPOTS_FILE,
)


# -----------------------------
# Configuration
# -----------------------------

DATA_DIRS = [
    INPUT_DIR / "WDPA_WDOECM_Mar2025_Public_AF_shp/WDPA_WDOECM_Mar2025_Public_AF_shp_0",
    INPUT_DIR / "WDPA_WDOECM_Mar2025_Public_AF_shp/WDPA_WDOECM_Mar2025_Public_AF_shp_1",
    INPUT_DIR / "WDPA_WDOECM_Mar2025_Public_AF_shp/WDPA_WDOECM_Mar2025_Public_AF_shp_2",
]

SHAPEFILES = [
    DATA_DIRS[0] / "WDPA_WDOECM_Mar2025_Public_AF_shp-polygons.shp",
    DATA_DIRS[1] / "WDPA_WDOECM_Mar2025_Public_AF_shp-polygons.shp",
    DATA_DIRS[2] / "WDPA_WDOECM_Mar2025_Public_AF_shp-polygons.shp",
]

TARGET_CRS = "EPSG:4326"

INDIAN_OCEAN_BOUNDS = {"min_lon": 30.0,
                       "max_lon": 70.0,
                       "min_lat": -35.0,
                       "max_lat": 5.0}

SAVEDIR = OUTPUT_DIR
file_MPA_SWIO = "MPA_SWIO_mask.nc"

SPOTS_FILE = Path(SPOTS_FILE)


# -----------------------------
# Core 
# -----------------------------

def load_shapefile(path: Path) -> gpd.GeoDataFrame:
    
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Empty shapefile: {path}")
    if gdf.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    return gdf.to_crs(TARGET_CRS)


def load_all_shapefiles(paths: List[Path]) -> gpd.GeoDataFrame:
    
    frames = [load_shapefile(p) for p in paths]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=TARGET_CRS)


def build_indian_ocean_polygon() -> gpd.GeoDataFrame:
    
    b = INDIAN_OCEAN_BOUNDS
    return gpd.GeoDataFrame(geometry=[box(b["min_lon"], b["min_lat"],
    				          b["max_lon"], b["max_lat"])], crs=TARGET_CRS)

def load_land_polygons() -> object:
    
    land_path = Path("../ne_110m_land/ne_110m_land.shp")
    if not land_path.exists():
        raise FileNotFoundError("Missing Natural Earth land file. "
            			"Download ne_110m_land.shp to connectivities/ne_110m_land/")
    land = gpd.read_file(land_path).to_crs(TARGET_CRS)
    return unary_union(land.geometry)


def filter_polygons_in_region(polygons: gpd.GeoDataFrame,
			      region: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
			      
    return polygons[polygons.intersects(region.geometry.iloc[0])].copy()


def keep_mostly_marine_polygons(polygons: gpd.GeoDataFrame,
        land_geometry, max_land_overlap: float = 0.1,
        ) -> gpd.GeoDataFrame:
    """
    Keep polygons with limited land overlap.

    Parameters
    ----------
    polygons : GeoDataFrame
        Input polygons.

    land_geometry : shapely geometry
        Unioned land geometry.

    max_land_overlap : float
        Maximum allowed fraction of polygon area overlapping land.
        Example:
            0.1 -> allows 10% land overlap.
    """

    marine_indices = []

    for idx, geom in polygons.geometry.items():

        if geom is None or geom.is_empty:
            continue

        polygon_area = geom.area

        if polygon_area == 0:
            continue

        land_intersection = geom.intersection(land_geometry)

        overlap_ratio = land_intersection.area / polygon_area

        if overlap_ratio < max_land_overlap:
            marine_indices.append(idx)

    return polygons.loc[marine_indices].copy()


def keep_only_marine_polygons(polygons: gpd.GeoDataFrame,
			      land_geometry) -> gpd.GeoDataFrame:
    
    # Marine polygons do NOT intersect land
    mask = ~polygons.intersects(land_geometry)
    return polygons.loc[mask].copy()
  

def plot_marine_polygons(polygons: gpd.GeoDataFrame,
                         points: list[tuple[float, float]] | None = None,
                         save: bool = False) -> None:
    
    projection = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 10),
                           subplot_kw={"projection": projection})

    ax.set_extent([INDIAN_OCEAN_BOUNDS["min_lon"],
                   INDIAN_OCEAN_BOUNDS["max_lon"],
                   INDIAN_OCEAN_BOUNDS["min_lat"],
                   INDIAN_OCEAN_BOUNDS["max_lat"]], crs=projection)

    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    # Gridlines
    gl = ax.gridlines(crs=projection, draw_labels=True,
                      linewidth=0.6, linestyle="--", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    # Plot marine polygons
    polygons.plot(ax=ax,
                  transform=projection,
                  facecolor="lightcoral",
                  edgecolor="black",
                  linewidth=0.6,
                  alpha=0.9)

    if points is not None and len(points) > 0:
        lons, lats = zip(*points)
        ax.scatter(lons, lats,
                   transform=projection,
                   s=60,
                   c="green",
                   marker="o",
                   edgecolor="black",
                   linewidth=0.8,
                   zorder=5,
                   label="Identified release points")

        ax.legend()

    ax.set_title("Marine Protected Areas (MPAs) in the SWIO", fontsize=14)
    plt.tight_layout()

    savepath = FIGURES_DIR / "map_MPA_SWIO.png"
    if save:
        os.makedirs(savepath.parent, exist_ok=True)
        if savepath.exists():
            print(f' \n!!WARNING: {savepath} already exists. Overriding...')
        print(f"Saving figure to {savepath}...")
        plt.savefig(savepath, dpi=300)
        plt.close(fig)
        print("Done!\n ")
    else:
        print('Displaying...')
        plt.show()
        plt.close(fig)


def polygons_to_xarray(polygons: gpd.GeoDataFrame,
                       resolution=0.1) -> xr.DataArray:
    
    b = INDIAN_OCEAN_BOUNDS
    
    lon = np.arange(b["min_lon"], b["max_lon"] + resolution, resolution)
    lat = np.arange(b["min_lat"], b["max_lat"] + resolution, resolution)

    nx = len(lon)
    ny = len(lat)

    transform = Affine.translation(lon[0], lat[0]) * Affine.scale(resolution, resolution)

    shapes = [(geom, 1) for geom in polygons.geometry]

    raster = features.rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    da = xr.DataArray(
        raster,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="marine_protected_area"
    )

    return da


def save_polygons_to_nc(polygons_da: xr.DataArray,
                        SAVEDIR, file) -> None:

    savedir_path = Path(SAVEDIR)
    os.makedirs(savedir_path, exist_ok=True)

    out_file = savedir_path / file
    print(f'\nSaving SWIO marine polygons to {out_file}...')
    polygons_da.to_netcdf(out_file)
    print('Done')


# -----------------------------
# CLI Arguments
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and plot Marine Protected Areas (MPAs) in the SWIO."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the figure as .png under figures/ instead of displaying it."
    )
    return parser.parse_args()


# -----------------------------
# Entry point
# -----------------------------

def main() -> None:
    args = parse_args()

    polygons = load_all_shapefiles(SHAPEFILES)
    indian_ocean = build_indian_ocean_polygon()
    land = load_land_polygons()
    ocean_polygons = filter_polygons_in_region(polygons, indian_ocean)
    
    # marine_polygons = keep_only_marine_polygons(ocean_polygons, land)
    marine_polygons = keep_mostly_marine_polygons(ocean_polygons, land, max_land_overlap=0.96)

    if marine_polygons.empty:
        raise RuntimeError("No marine polygons found.")
    else: 
        mpa_da = polygons_to_xarray(marine_polygons, resolution=0.1)
        save_polygons_to_nc(mpa_da, SAVEDIR, file_MPA_SWIO)
 
    release_points = [tuple(r) for r in np.loadtxt(SPOTS_FILE, comments='#')]
     
    #plot_marine_polygons(marine_polygons, save=args.save)
    plot_marine_polygons(marine_polygons, points=release_points, save=args.save)

if __name__ == "__main__":
    main()

