import time
import warnings

warnings.filterwarnings('ignore', '.*area.*')
from cutil import *
import requests
import concurrent.futures
import io
from typing import Iterator

import numpy as np
import pandas as pd
import datashader as ds
import pyarrow as pa
import spatialpandas as sp
from datashader.core import bypixel
import pygeos.creation
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
import math

import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

invtransformer = Transformer.from_crs(4326,3395)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def get_flat_coords_offset_arrays(arr):
    """
    Version for MultiPolygon data
    """
    # explode/flatten the MultiPolygons
    arr_flat, part_indices = pygeos.get_parts(arr, return_index=True)
    # the offsets into the multipolygon parts
    offsets1 = np.insert(np.bincount(part_indices).cumsum(), 0, 0)

    # explode/flatten the Polygons into Rings
    arr_flat2, ring_indices = pygeos.geometry.get_rings(arr_flat, return_index=True)
    # the offsets into the exterior/interior rings of the multipolygon parts 
    offsets2 = np.insert(np.bincount(ring_indices).cumsum(), 0, 0)

    # the coords and offsets into the coordinates of the rings
    coords, indices = pygeos.get_coordinates(arr_flat2, return_index=True)
    offsets3 = np.insert(np.bincount(indices).cumsum(), 0, 0)
    
    return coords, offsets1, offsets2, offsets3

def spatialpandas_from_pygeos(arr):
    coords, offsets1, offsets2, offsets3 = get_flat_coords_offset_arrays(arr)
    coords_flat = coords.ravel()
    offsets3 *= 2
    
    # create a pyarrow array from this
    _parr3 = pa.ListArray.from_arrays(pa.array(offsets3), pa.array(coords_flat))
    _parr2 = pa.ListArray.from_arrays(pa.array(offsets2), _parr3)
    parr = pa.ListArray.from_arrays(pa.array(offsets1), _parr2)
    
    return sp.geometry.MultiPolygonArray(parr)

def polygons(self, source, geometry, agg=None):
    from datashader.glyphs import PolygonGeom
    from datashader.reductions import any as any_rdn
    from spatialpandas import GeoDataFrame
    from spatialpandas.dask import DaskGeoDataFrame
    if isinstance(source, DaskGeoDataFrame):
        # Downselect partitions to those that may contain polygons in viewport
        x_range = self.x_range if self.x_range is not None else (None, None)
        y_range = self.y_range if self.y_range is not None else (None, None)
        source = source.cx_partitions[slice(*x_range), slice(*y_range)]
    elif isinstance(source, gpd.GeoDataFrame):
        # Downselect actual rows to those for which the polygon is in viewport
        x_range = self.x_range if self.x_range is not None else (None, None)
        y_range = self.y_range if self.y_range is not None else (None, None)
        source = source.cx[slice(*x_range), slice(*y_range)]
        # Convert the subset to ragged array format of spatialpandas
        geometries = spatialpandas_from_pygeos(source.geometry.array.data)
        source = pd.DataFrame(source)
        source["geometry"] = geometries
    elif not isinstance(source, GeoDataFrame):
        raise ValueError(
            "source must be an instance of spatialpandas.GeoDataFrame or \n"
            "spatialpandas.dask.DaskGeoDataFrame.\n"
            "  Received value of type {typ}".format(typ=type(source)))

    if agg is None:
        agg = any_rdn()
    glyph = PolygonGeom(geometry)
    return bypixel(source, self, glyph, agg)

ds.Canvas.polygons = polygons
cvs = ds.Canvas(plot_width=256, plot_height=256)

def elevation(filtered, bbox):
    proxy = pd.DataFrame({'min_height': 0, 'height': 0, 'geometry': bbox}, index=[len(filtered)])
    proxy = gpd.GeoDataFrame(proxy)
    proxy.crs = '3395'

    clipped = gpd.clip(filtered, proxy)
    intersection = pd.concat([proxy, clipped], ignore_index=True)
    intersection = intersection[intersection.geom_type.isin(['Polygon', 'MultiPolygon'])]
    if len(intersection) > 0:
#         intersection = sp.GeoDataFrame(intersection)
        values = cvs.polygons(intersection, geometry='geometry', agg=ds.max("height"))
    else:
        values = np.zeros((256,256))
    values = np.flipud(values)
    return values

def compute_tile(gdf, i, j, zoom, max_height):
    bb0 = num2deg(i,j,zoom)
    bb1 = num2deg(i+1,j+1,zoom)
    bb0 = invtransformer.transform(bb0[0],bb0[1])
    bb1 = invtransformer.transform(bb1[0],bb1[1])
    bbox = box(bb0[0],bb0[1],bb1[0],bb1[1])
#     filtered = gdf.cx[bb0[0]:bb1[0],bb0[1]:bb1[1]]
    filtered = gdf.loc[gdf.sindex.intersection(bbox.bounds)]

    if len(filtered) > 0:
        values = elevation(filtered, bbox)
        return 255.0 * (values / max_height)
        # create_image(values, i, j, zoom, max_height, outputfolder)

def _get_image(xtile: int, ytile: int, zoom: int, file, max_height: float) -> np.ndarray:
    gdf = gpd.read_file(file)
    gdf = gdf.to_crs(3395)
    image = compute_tile(gdf, xtile, ytile, zoom, max_height)

    return image


def get_images(
        xtiles: list[int], ytiles: list[int], zoom: int, max_height: float
) -> dict[tuple[int, int], np.ndarray]:
    urls = (
        f'http://data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{xtile}/{ytile}.json'
        for xtile, ytile in zip(xtiles, ytiles)
    )

    def func(url, x, y, session: requests.Session):
        response = session.get(url)
        # response = requests.get(url)
        response.raise_for_status()
        return io.StringIO(response.text), x, y

    with concurrent.futures.ThreadPoolExecutor() as pool, requests.Session() as session:
        futures = {
            pool.submit(func, url, xtile, ytile, session)
            for url, xtile, ytile in zip(urls, xtiles, ytiles)
        }
        results: Iterator[tuple[io.StringIO, int, int]] = (
            future.result()
            for future in concurrent.futures.as_completed(futures)
        )
        images = {
            (x, y): _get_image(x, y, zoom, response, max_height)
            for response, x, y in results
        }
    return images


def display_images(
        xtiles: list[int | float], ytiles: list[int | float], zoom: int, max_height: float
) -> None:
    images = get_images(xtiles, ytiles, zoom, max_height)
    for image in images.values():
        plt.figure()
        plt.imshow(image)


def get_image(xtile: int, ytile: int, zoom: int, max_height: float) -> np.ndarray:
    if isinstance(xtile, float):
        xtile, ytile = deg2num(xtile, ytile, zoom, True)
    url = f'http://data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{xtile}/{ytile}.json'
    t = time.time()
    file = io.StringIO(requests.get(url).text)
    print(f"http request takes {(time.time() - t)} seconds")
    return _get_image(xtile, ytile, zoom, file, max_height)


def display_image(xtile: int, ytile: int, zoom: int, max_height: float) -> None:
    image = get_image(xtile, ytile, zoom, max_height)
    plt.figure()
    plt.imshow(image)


class Namespace:
    xtile: list[int]
    ytile: list[int]
    zoom: int
    max: float


if __name__ == '__main__':
    places = [
        (40.740530392418506, -73.99426405495954),
        (40.823631367500454, -73.93789016206937),
        (40.6148254758228, -74.03463693864036),
    ]
    tiles = [
        deg2num(*place[::-1], 15, True)
        for place in places
    ]
    xtiles = [tile[0] for tile in tiles]
    ytiles = [tile[1] for tile in tiles]
    display_images(xtiles, ytiles, 15, 300)
    # %%
    # get_image(*(40.74446651502901, -74.0023459850658)[::-1], 15, 500)
