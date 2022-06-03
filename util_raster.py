from pyproj import CRS
import math
import tempfile
from typing import Iterator

import rasterio
import rasterio.crs

import concurrent.futures
import os

import pyproj

from shadow.cutil import deg2num, num2deg
import cv2
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


def get_shadow_image(
        gw: float,
        gs: float,
        ge: float,
        gn: float,
        zoom: int,
        basedir: str,
        threshold: tuple[float, float]
) -> np.ndarray:
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
    xtiles = np.char.add(xtiles, os.sep)

    ytiles = np.repeat(ytiles, c_tilecount)
    xtiles = np.tile(xtiles, r_tilecount)

    tiles = np.char.add(xtiles, ytiles)
    directory = os.path.join(basedir, str(zoom)) + os.sep
    directory = os.path.normpath(directory)
    if not directory.endswith(os.sep):
        directory += os.sep
    relevant = np.char.add(directory, tiles)
    paths = [
        path
        for path in relevant
        if os.path.exists(path)
    ]
    images: Iterator[np.ndarray] = concurrent.futures.ThreadPoolExecutor().map(lambda p: cv2.imread(p)[:, :, 0], paths)
    partitions: Iterator[str] = (
        path.rpartition('.')[0]
        for path in paths
    )
    partitions: Iterator[tuple[str, str, str]] = (
        partition.rpartition(os.sep)
        for partition in partitions
    )
    xtiles_ytiles: Iterator[tuple[int, int]] = (
        (
            int(partition[0].rpartition(os.sep)[2]),
            int(partition[2])
        )
        for partition in partitions
    )

    raster = np.zeros((256 * r_tilecount, 256 * c_tilecount), dtype=np.uint8)
    for (xtile, ytile), image in zip(xtiles_ytiles, images):
        raster[rslices[ytile], cslices[xtile]] = image

    floor = math.floor(threshold[0] * 255)
    ceil = math.ceil(threshold[1] * 255)
    raster[np.logical_or(
        raster < floor,
        raster >= ceil
    )] = -1
    # if threshold > 0:
    #     cutoff = math.ceil(threshold * 255)
    #     image[np.logical_and(0 <= image, image < cutoff)] = nodata

    return raster


def get_raster_path(
        gw: float,
        gs: float,
        ge: float,
        gn: float,
        zoom: int,
        basedir: str,
        threshold: tuple[float, float] = (0.0, 1.0),
        outdir: str = None,
        nodata: int = -1
) -> str:
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)
    te += 1
    ts += 1

    if outdir is None:
        outdir = tempfile.gettempdir()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, f'{zoom}_{tw}_{ts}_{te}_{tn}.tif')

    image = get_shadow_image(
        gw,
        gs,
        ge,
        gn,
        zoom=zoom,
        basedir=basedir,
        threshold=threshold,
    )
    height, width = image.shape
    # width, height = image.shape

    gw, gs, = num2deg(tw, ts, zoom, True)
    ge, gn = num2deg(te, tn, zoom, True)

    transform = rasterio.transform.from_bounds(gw, gs, ge, gn, width, height)

    with rasterio.open(
            outpath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.int16,
            nodata=nodata,
            crs=4326,
            transform=transform,
    ) as f:
        f.write(image, 1)

    return outpath


def get_raster_affine(
        gw: float,
        gs: float,
        ge: float,
        gn: float,
        zoom: int,
        basedir: str,
        threshold: tuple[float, float] = (0.0, 1.0)
) -> tuple[np.ndarray, tuple]:
    "Returns the array and the Affnie transformation"
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)
    te += 1
    ts += 1

    # if outdir is None:
    #     outdir = tempfile.gettempdir()
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    # # outpath = os.path.join(outdir, f'{zoom}_{tw}_{ts}_{te}_{tn}.tif')

    image = get_shadow_image(
        gw,
        gs,
        ge,
        gn,
        zoom=zoom,
        basedir=basedir,
        threshold=threshold,
    )
    height, width = image.shape
    gw, ts = num2deg(tw, ts, zoom, True)
    ge, tn = num2deg(te, tn, zoom, True)

    image = image / 255
    transform = rasterio.transform.from_bounds(gw, gs, ge, gn, width, height)
    return image, transform


if __name__ == '__main__':
    tiles = get_shadow_image(
        *(40.702844950247666, -74.02244810805952)[::-1],
        *(40.78270102430847, -73.93524412566495)[::-1],
        16,
        '/home/arstneio/Downloads/shadows/test/16-winter/'
    )

    print()
    # cells = get_cells_from_tiles(tiles, os.path.join('/home/arstneio/Downloads/shadows/nyc-sep-22/', str(16)))
