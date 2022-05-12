if True:
    # TODO: For some strange reason, importing geopandas before shadow.cutil causes an ImportError
    from utils import (
        load_image,
        deg2num,
        nums2degs,
        num2deg
    )

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import cv2
import dask_geopandas
import geopandas as gpd
import geopy.distance
import numpy as np
import pandas as pd
import pygeos.creation
import pygeos.creation
import pyproj
from geopandas import GeoDataFrame
from pandas import Series
from pyproj import Transformer

from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed
import contextlib
import joblib
import time
from tqdm import tqdm

def get_tiles(gdf: GeoDataFrame, zoom: int) -> tuple[GeoDataFrame, GeoDataFrame]:
    # TODO: dask.delayed to speedup this thing
    # Get tile bounds from GDF
    pw, ps, pe, pn = gdf.total_bounds
    trans = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    gw, gn = trans.transform(pw, pn)
    ge, gs = trans.transform(pe, ps)

    tw, tn = deg2num(gw, gn, zoom, always_xy=True)
    te, ts = deg2num(ge, gs, zoom, always_xy=True)

    # Just making sure that the tiles are actually north, west
    tn, ts = min(tn, ts), max(tn, ts)
    tw, te = min(tw, te), max(tw, te)

    # np.ndarray indexing is [row, column], so I am using [north, west] to maintain that convention
    # Convention: repeat rows, tile columns

    # Slippy Tiles
    tn = np.arange(tn, ts, dtype=np.uint)  # xtile goes from n to s
    tw = np.arange(tw, te, dtype=np.uint)  # ytile goes from w to e

    # Geographic
    # Generate from northmost tiles and westmost tiles O(n) instead of all tiles O(n^2)
    _, tgn = nums2degs(np.repeat(tw[0], len(tn)), tn, zoom, always_xy=True)
    tgw, _ = nums2degs(tw, np.repeat(tn[0], len(tw)), zoom, always_xy=True)
    tgs = np.append(
        tgn[1:],
        num2deg(tw[0], ts, zoom, always_xy=True)[1]
    )
    tge = np.append(
        tgw[1:],
        num2deg(te, tn[0], zoom, always_xy=True)[0]
    )

    # Projected
    # Generate from northmost geographic and westmost geographic O(n) instead of all tiles O(n^2)
    trans = Transformer.from_crs(4326, gdf.crs, always_xy=True)
    _, tpn = trans.transform(np.repeat(tgw[0], len(tgn)), tgn)
    tpw, _ = trans.transform(tgw, np.repeat(tgn[0], len(tgw)))
    tps = np.append(
        tpn[1:],
        trans.transform(tgw[0], tgs[-1])[1]
    )
    tpe = np.append(
        tpw[1:],
        trans.transform(tge[-1], tgn[0])[0]
    )

    repeat_rows = len(tw)
    tile_columns = len(tn)
    tn = np.repeat(tn, repeat_rows)
    tw = np.tile(tw, tile_columns)

    tns = tn << 32
    tntw = np.bitwise_or(tns, tw)
    tntw = pd.Index(tntw, name='tntw', dtype=np.uint64)

    tpw = np.tile(tpw, tile_columns)
    tps = np.repeat(tps, repeat_rows)
    tpe = np.tile(tpe, tile_columns)
    tpn = np.repeat(tpn, repeat_rows)
    geometry = pygeos.creation.box(tpw, tps, tpe, tpn)
    h = (tps - tpn)
    w = (tpe - tpw)

    tiles = GeoDataFrame({
        # 'tn': tn, 'tw': tw,
        'tpw': tpw, 'tps': tps, 'tpe': tpe, 'tpn': tpn,
        'h': h, 'w': w,
    }, index=tntw, geometry=geometry, crs=gdf.crs)

    itile, igdf = gdf.sindex.query_bulk(tiles.geometry)
    gdf: GeoDataFrame = gdf.iloc[igdf]
    tiles: GeoDataFrame = tiles.iloc[itile]
    gdf = gdf.set_index(tiles.index)
    # tiles: GeoDataFrame = tiles.loc[tiles.index.unique()]
    tiles: GeoDataFrame = tiles.loc[~tiles.index.duplicated()]
    return gdf, tiles


def get_cells(tiles: GeoDataFrame, cell_length: float = 10.0) -> GeoDataFrame:
    s, w, n, e = tiles.geometry.iloc[0].bounds
    trans = pyproj.Transformer.from_crs(tiles.crs, 4326, always_xy=True)
    (w, e), (s, n) = trans.transform((w, e), (n, s))
    distance = geopy.distance.distance((n, w), (s, w)).meters

    length_cells = math.ceil(distance / cell_length)
    tile_count = len(tiles)
    area_cells = length_cells ** 2

    dh = tiles['h'].values / length_cells
    dw = tiles['w'].values / length_cells

    # cn = np.repeat(range(length_cells), length_cells)
    # cw = np.tile(range(length_cells), length_cells)
    # cs = np.repeat(range(1, length_cells + 1), length_cells)
    # ce = np.tile(range(1, length_cells + 1), length_cells)
    cn = np.repeat(
        np.arange(length_cells, dtype=np.uint64), length_cells,
    )
    cw = np.tile(
        np.arange(length_cells, dtype=np.uint64), length_cells,
    )
    cs = np.repeat(
        np.arange(1, length_cells + 1, dtype=np.uint64), length_cells
    )
    ce = np.tile(
        np.arange(1, length_cells + 1, dtype=np.uint64), length_cells
    )

    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)
    csr = np.tile(cs, tile_count)
    cer = np.tile(ce, tile_count)

    tpnr = np.repeat(tiles['tpn'].values, area_cells)
    tpwr = np.repeat(tiles['tpw'].values, area_cells)

    dhr = np.repeat(dh, area_cells)
    dwr = np.repeat(dw, area_cells)

    cpn = tpnr + (dhr * cnr)
    cps = tpnr + (dhr * csr)
    cpw = tpwr + (dwr * cwr)
    cpe = tpwr + (dwr * cer)

    index = tiles.index.repeat(area_cells)
    geometry = pygeos.creation.box(cpw, cps, cpe, cpn)

    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)

    area = np.abs(dh * dw)
    arear = np.repeat(area, area_cells)

    cells = GeoDataFrame({
        # 'tn': tnr, 'twr': twr,
        'cn': cnr, 'cw': cwr,
        'area': arear,
    }, index=index, geometry=geometry, crs=tiles.crs)
    return cells


def partition_mapping(
        cells: GeoDataFrame,
        gdf: GeoDataFrame,
        max_height: float,
        directory: str,
        cell_length: int,
        zoom:int,
):
    # I don't care about chained assignment because after this is done the GDFs are just going to be thrown in the trash
    gdf = gdf.loc[cells.index.unique()]  # Get only the geometry relevant to the cells
    icell, igdf = gdf.sindex.query_bulk(cells.geometry)

    cells: GeoDataFrame = cells.iloc[icell]
    gdf: GeoDataFrame = gdf.iloc[igdf]

    intersection: Series = cells.intersection(gdf.geometry, align=False).area

    # weight: Series = intersection.values / cells['area'].values * gdf['height'].values / max_height * (2 ** 16 - 1)
    weight: np.ndarray = (
            intersection.values
            / cells['area'].values
            * gdf['height'].values
            / max_height
    )
    cells['weight'] = weight
    agg: GeoDataFrame = cells.groupby(['tntw', 'cn', 'cw'], sort=False).agg({'weight': 'sum'})
    agg['weight'] = (
            agg.values * (2 ** 16 - 1)
    ).astype(np.uint16)

    groups = agg.groupby('tntw', sort=False).groups
    tntw = np.fromiter(groups.keys(), dtype=np.uint64)
    tn = np.bitwise_and(tntw, (2 ** 64 - (2 ** 32))) >> 32
    tw = np.bitwise_and(tntw, (2 ** 32 - 1))

    paths = [
        os.path.join(directory, f'{zoom}/{tw_}/{tn_}.png')
        for tn_, tw_ in zip(tn, tw)
    ]
    nodirs = (
        dir
        for path in paths
        if not os.path.exists(dir := os.path.dirname(path))
    )
    with ThreadPoolExecutor() as te:
        te.map(os.makedirs, nodirs)
    subaggs: Iterator[Series] = (
        agg.loc[loc]
        for loc in groups.values()
    )
    images = (
        load_image(
            cn=subagg.index.get_level_values('cn').values,
            cw=subagg.index.get_level_values('cw').values,
            weights=subagg['weight'].values,
            cell_length=cell_length,
        )
        for subagg in subaggs
    )
    with ThreadPoolExecutor() as te:
        te.map(cv2.imwrite, paths, images)

def run(
        gdf: GeoDataFrame,
        zoom: int,
        max_height: float = None,
        outputfolder: str = None,
) -> None:
    # TODO: I think Los Angeles will fail because of a huge cellsize. I need to look into implementing cells with
    #   from_delayed
    """
    :param gdf: building data with column 'height'
    :param zoom: slippy tile zoom
    :param max_height: maximum height across datasets, for normalization purposes
    :param outputfolder: output directory
    :return:
    """
    if outputfolder is None:
        outputfolder = os.getcwd()
    if max_height is None:
        max_height = gdf['height'].max()
    gdf, tiles = get_tiles(gdf, zoom)
    cells = get_cells(tiles, 10.0)
    cell_length = len(cells['cn'].unique())
    grid_size = cell_length ** 2
    chunksize = grid_size * 50
    cells: dask_geopandas.GeoDataFrame = dask_geopandas.from_geopandas(cells, chunksize=chunksize, sort=True)
    gdf: dask_geopandas.GeoDataFrame = dask_geopandas.from_geopandas(gdf, chunksize=chunksize, sort=True)

    pd.set_option('mode.chained_assignment', None)
    cells.map_partitions(
        partition_mapping,
        gdf=gdf,
        meta=(None, None),
        max_height=max_height,
        directory=outputfolder,
        zoom=zoom,
        cell_length=cell_length,
        align_dataframes=True,
    ).compute()
    pd.set_option('mode.chained_assignment', 'warn')