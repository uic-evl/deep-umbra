import rasterio.crs
import pyproj
import rasterio
from rasterio.transform import Affine
import functools
import glob
import tempfile
from typing import Iterator

if True:
    from shadow.cutil import (
        xtiles_from_lons,
        ytiles_from_lats,
        lons_from_xtiles,
        lats_from_ytiles,
    )
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import geopandas as gpd

import skimage.io
import concurrent.futures
import os

import pandas as pd
import pyproj

from cutil import deg2num, nums2degs, num2deg
import pygeos.creation
import cv2
import cython
import numpy as np

import pyproj.aoi


def get_utm_from_lon_lat(lon: float, lat: float) -> pyproj.crs.CRS:
    buffer = .001
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name='WGS 84',
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon - buffer,
            south_lat_degree=lat - buffer,
            east_lon_degree=lon + buffer,
            north_lat_degree=lat + buffer,
        ),
    )
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def get_shadow_image(gw: float, gs: float, ge: float, gn: float, zoom: int, basedir: str) -> np.ndarray:
    # Note: python 3.9 glob.glob() does not have kwarg root_dir
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)

    ytiles = np.arange(tn, ts + 1, dtype=np.uint32)
    xtiles = np.arange(tw, te + 1, dtype=np.uint32)

    r_tilecount = len(ytiles)
    c_tilecount = len(xtiles)

    cslices = {
        xtile: slice(l, l + 256)
        for l, xtile in zip(range(0, 256 * c_tilecount, 256), xtiles)
    }
    rslices = {
        ytile: slice(l, l + 256)
        for l, ytile in zip(range(0, 256 * r_tilecount, 256), ytiles)
    }

    ytiles = ytiles.astype('U10')
    xtiles = xtiles.astype('U10')

    ytiles = np.char.add(ytiles, '.png')
    xtiles = np.char.add(xtiles, '/')

    ytiles = np.repeat(ytiles, c_tilecount)
    xtiles = np.tile(xtiles, r_tilecount)

    tiles = np.char.add(xtiles, ytiles)
    tiles = set(tiles)

    pathname = os.path.join(basedir, f'**/{zoom}/[0-9]*/[0-9]*.png')

    # paths is not aligned with xtiles
    paths: list[str] = [
        path
        for path in glob.iglob(pathname, recursive=True)
        if path[
           path.rfind('/', 0, path.rfind('/')) + 1::
           ] in tiles
    ]
    images: Iterator[np.ndarray] = concurrent.futures.ThreadPoolExecutor().map(lambda p: cv2.imread(p)[:, :, 0], paths)
    partitions: Iterator[str] = (
        path.rpartition('.')[0]
        for path in paths
    )
    partitions: Iterator[tuple[str, str, str]] = (
        partition.rpartition('/')
        for partition in partitions
    )
    xtiles_ytiles: Iterator[tuple[int, int]] = (
        (
            int(partition[0].rpartition('/')[2]),
            int(partition[2])
        )
        for partition in partitions
    )

    raster = np.zeros((256 * r_tilecount, 256 * c_tilecount), dtype=np.uint8)
    for (xtile, ytile), image in zip(xtiles_ytiles, images):
        raster[rslices[ytile], cslices[xtile]] = image
    return raster


def get_raster_path(gw: float, gs: float, ge: float, gn: float, zoom: int, basedir: str, outdir: str = None) -> str:
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)
    te += 1
    ts += 1

    if outdir is None:
        outdir = tempfile.gettempdir()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, f'{zoom}_{tw}_{ts}_{te}_{tn}.tif')
    if os.path.exists(outpath):
        return outpath

    image = get_shadow_image(gw, gs, ge, gn, zoom, basedir)
    height, width = image.shape
    gw, ts = num2deg(tw, ts, zoom, True)
    ge, tn = num2deg(te, tn, zoom, True)

    transform = rasterio.transform.from_bounds(gw, gs, ge, gn, width, height)

    with rasterio.open(
            outpath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=image.dtype,
            nodata=0,
            crs='+proj=latlong',
            transform=transform,
    ) as f:
        f.write(image, 1)

    return outpath


if __name__ == '__main__':
    tiles = get_shadow_image(
        *(40.702844950247666, -74.02244810805952)[::-1],
        *(40.78270102430847, -73.93524412566495)[::-1],
        16,
        # '/home/arstneio/Downloads/shadows/nyc-sep-22/'
        '/home/arstneio/Downloads/shadows/'
    )
    path = get_raster_path(
        *(40.702844950247666, -74.02244810805952)[::-1],
        *(40.78270102430847, -73.93524412566495)[::-1],
        16,
        # '/home/arstneio/Downloads/shadows/nyc-sep-22/'
        '/home/arstneio/Downloads/shadows/'
    )
    print()
    # cells = get_cells_from_tiles(tiles, os.path.join('/home/arstneio/Downloads/shadows/nyc-sep-22/', str(16)))
