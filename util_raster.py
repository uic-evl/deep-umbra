import concurrent.futures
import math
import os
import tempfile
from typing import Iterator, Union, Optional

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pyproj.aoi
import rasterio
import rasterio.plot
import rasterio.plot
from geopandas import GeoDataFrame


def _deg2num(lon_deg, lat_deg, zoom, always_xy, floored: bool):
    # lat_rad = math.radians(lat_deg)
    lat_rad = lat_deg * math.pi / 180.0

    n = 2 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    if floored:
        xtile = int(xtile)
        ytile = int(ytile)
    if always_xy:
        return xtile, ytile
    else:
        return ytile, xtile


def deg2num(
        lon_deg: float,
        lat_deg: float,
        zoom: int,
        always_xy=True,
        floored=False,
) -> tuple[float, float]:
    """

    :param lat_deg:
    :param lon_deg:
    :param zoom:
    :return: xtile, ytile
    """
    return _deg2num(lon_deg, lat_deg, zoom, always_xy, floored)


def _num2deg(xtile, ytile, zoom, always_xy):
    n = 2 ** zoom
    lon_deg = xtile / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    # lat_deg = math.degrees(lat_rad)
    lat_deg = lat_rad * 180.0 / math.pi
    if always_xy:
        return lon_deg, lat_deg
    else:
        return lat_deg, lon_deg
    # return lat_deg, lon_deg


def num2deg(xtile: float, ytile: float, zoom: int, always_xy=False) -> tuple[float, float]:
    """

    :param xtile:
    :param ytile:
    :param zoom:
    :return: latitude, longitude
    """
    return _num2deg(xtile, ytile, zoom, always_xy)


def _xtiles_from_lons(lons, zoom, ):
    length = len(lons)
    n = 2 ** zoom
    xtiles = np.zeros(length, dtype=np.uint32)
    for k in range(length):
        xtiles[k] = ((lons[k] + 180) / 360 * n)
    return xtiles


def xtiles_from_lons(lons: np.ndarray, zoom: int):
    return _xtiles_from_lons(lons, zoom)


def _ytiles_from_lats(lons, zoom, ):
    length = len(lons)
    n = 2 ** zoom
    ytiles = np.zeros(length, dtype=np.uint32)
    for k in range(length):
        ytiles[k] = ((lons[k] + 180) / 360 * n)
    return ytiles


def ytiles_from_lats(lats: np.ndarray, zoom: int):
    return _ytiles_from_lats(lats, zoom)


def _lons_from_xtiles(xtiles, zoom):
    length = len(xtiles)
    n = 2 ** zoom
    lons = np.zeros(length, dtype=np.float64)
    for k in range(length):
        lons[k] = 360.0 * xtiles[k] / n - 180
    return lons


def lons_from_xtiles(xtiles: np.ndarray, zoom: int):
    return _lons_from_xtiles(xtiles, zoom)


def _lats_from_ytiles(ytiles, zoom):
    length = len(ytiles)
    n = 2 ** zoom
    lats = np.zeros(length, dtype=np.float64)
    rad_to_deg = 180.0 / math.pi
    for k in range(length):
        lats[k] = (
                math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytiles[k] / n))) * rad_to_deg
        )
    return lats


def lats_from_ytiles(ytiles: np.ndarray, zoom: int):
    return _lats_from_ytiles(ytiles, zoom)


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


def get_raster_size(
        gw: float,
        gs: float,
        ge: float,
        gn: float,
        zoom: int,
        mask: Optional[list[float]] = None,
) -> int:
    """

    :param gw: minx
    :param gs: miny
    :param ge: maxx
    :param gn: maxy
    :param zoom: slippy tile zoom
    :param mask: [miny, minx, maxy, maxx]
    :return: size of raster to be generated
    """
    if mask is not None:
        gs = max(gs, mask[0])
        gw = max(gw, mask[1])
        gn = min(gn, mask[2])
        ge = min(ge, mask[3])

    tw, tn = deg2num(gw, gn, zoom, True, floored=True)
    te, ts = deg2num(ge, gs, zoom, True, floored=True)

    ts += 1
    te += 1

    tiles = (te - tw) * (ts - tn)
    # TODO: if tile cells or dtype changes this must too
    cells = 65536 * tiles
    # 2 bytes per int16 cell, if the dtype changes so must this
    bytes: int = cells * 2
    return bytes


def get_shadow_image(
        gw: float,
        gs: float,
        ge: float,
        gn: float,
        zoom: int,
        basedir: str,
        mask: Optional[list[float]] = None,
        threshold: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    print(gw, gs, ge, gn, zoom, basedir, mask, threshold)
    if mask is not None:
        gs = max(gs, mask[0])
        gw = max(gw, mask[1])
        gn = min(gn, mask[2])
        ge = min(ge, mask[3])
    # Note: python 3.9 glob.glob() does not have kwarg root_dir
    tw, tn = deg2num(gw, gn, zoom, True, floored=True)
    te, ts = deg2num(ge, gs, zoom, True, floored=True)

    ytiles = np.arange(tn, ts + 1, dtype=np.uint32)
    xtiles = np.arange(tw, te + 1, dtype=np.uint32)

    r_tilecount = len(ytiles)
    c_tilecount = len(xtiles)

    # print(r_tilecount, c_tilecount)
    # return

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
        os.path.normpath(path.rpartition('.')[0])
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

    raster = np.zeros((256 * r_tilecount, 256 * c_tilecount), dtype=np.int16)
    for (xtile, ytile), image in zip(xtiles_ytiles, images):
        raster[rslices[ytile], cslices[xtile]] = image

    floor = math.floor(threshold[0] * 255)
    ceil = math.ceil(threshold[1] * 255)
    raster[np.logical_or(
        raster < floor,
        raster > ceil
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
        outpath: str = None,
        mask: Optional[list[float]] = None,
        nodata: int = -1
) -> str:
    if mask is not None:
        gs = max(gs, mask[0])
        gw = max(gw, mask[1])
        gn = min(gn, mask[2])
        ge = min(ge, mask[3])
    tw, tn = deg2num(gw, gn, zoom, True, floored=True)
    te, ts = deg2num(ge, gs, zoom, True, floored=True)
    te += 1
    ts += 1

    if outpath is None:
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


#
# def get_raster_affine(
#         gw: float,
#         gs: float,
#         ge: float,
#         gn: float,
#         zoom: int,
#         basedir: str,
#         threshold: tuple[float, float] = (0.0, 1.0)
# ) -> tuple[np.ndarray, tuple]:
#     "Returns the array and the Affnie transformation"
#     tw, tn = deg2num(gw, gn, zoom, True)
#     te, ts = deg2num(ge, gs, zoom, True)
#     te += 1
#     ts += 1
#
#     # if outdir is None:
#     #     outdir = tempfile.gettempdir()
#     # if not os.path.exists(outdir):
#     #     os.makedirs(outdir)
#     # # outpath = os.path.join(outdir, f'{zoom}_{tw}_{ts}_{te}_{tn}.tif')
#
#     image = get_shadow_image(
#         gw,
#         gs,
#         ge,
#         gn,
#         zoom=zoom,
#         basedir=basedir,
#         threshold=threshold,
#     )
#     height, width = image.shape
#     gw, ts = num2deg(tw, ts, zoom, True)
#     ge, tn = num2deg(te, tn, zoom, True)
#
#     image = image / 255
#     transform = rasterio.transform.from_bounds(gw, gs, ge, gn, width, height)
#     return image, transform


def overlay(
        gdf: Union[GeoDataFrame, str],
        raster: str,
        figsize: tuple = (15, 15),
        statistic: str = 'weighted',
        **kwargs,
) -> None:
    """
    Plot an overlay of the Surface GDF and shadow raster file
    :param gdf: GeoDataFrame or path to a GeoDataFrame
    :param raster: raster file
    :param figsize: ax figsize
    :param statistic: the statistic to plot ("weighted" is from sum/count)
    :param kwargs: kwargs to pass to plt.plot()
    :return: None
    """

    if isinstance(gdf, str):
        gdf = gpd.read_feather(gdf)
    if statistic == 'weighted' and 'weighted' not in gdf:
        gdf['weighted'] = (
                pd.Series.astype(gdf['sum'], float)
                / (gdf['count'] + gdf['nodata'])
        ).astype(float)
    raster = rasterio.open(raster)
    fig, ax = plt.subplots(figsize=figsize)
    rasterio.plot.show(raster, ax=ax)
    gdf.plot(column=statistic, cmap='rainbow', ax=ax, )


if __name__ == '__main__':
    # tiles = get_shadow_image(
    #     *(40.702844950247666, -74.02244810805952)[::-1],
    #     *(40.78270102430847, -73.93524412566495)[::-1],
    #     16,
    #     '/home/arstneio/Downloads/shadows/test/16-winter/'
    # )

    print()
    # cells = get_cells_from_tiles(tiles, os.path.join('/home/arstneio/Downloads/shadows/nyc-sep-22/', str(16)))
