import abc
import concurrent.futures
import concurrent.futures
import multiprocessing
import warnings
from typing import Type, Union, Optional, Iterable, Collection
from weakref import WeakKeyDictionary

import geopandas as gpd
import numpy as np
import rasterstats
import shapely.geometry.base
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import Series, DataFrame

from util_raster import get_utm_from_lon_lat, get_raster_path, get_raster_size, overlay

warnings.filterwarnings('ignore', '.*PyGEOS.*')

import osmium

import pandas as pd

import functools
import itertools
import os
import math
import tempfile
from typing import Iterator

import pyrosm


def osmium_extract(
        file: str,
        osmium_executable_path: str,
        bbox: list[float],
        bbox_latlon=True
) -> str:
    if bbox_latlon:
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    string: str = ','.join(str(b) for b in bbox)
    name = file.rpartition(os.sep)[2] if os.sep in file else file
    tempdir = tempfile.gettempdir()
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    temp = os.path.join(tempdir, name)
    os.system(f'{osmium_executable_path} extract -b {string} {file} -o {temp} --overwrite')
    return temp


@functools.singledispatch
def pyrosm_extract(
        source: str,
        osmium_executable_path: str = None,
        bbox: Optional[list[float]] = None,
        bbox_latlon=True
) -> str:
    path = pyrosm.get_data(source)
    if bbox:
        path = osmium_extract(path, osmium_executable_path, bbox, bbox_latlon)
    return path


@pyrosm_extract.register(list)
def _(
        source: list[str],
        osmium_executable_path: str = None,
        bbox: Union[list[float], None] = None,
        bbox_latlon=True
) -> Iterator[str]:
    with concurrent.futures.ThreadPoolExecutor() as threads, concurrent.futures.ProcessPoolExecutor() as processes:
        files = threads.map(pyrosm.get_data, source)
        if bbox is not None:
            yield from processes.map(
                osmium_extract,
                itertools.repeat(osmium_executable_path, len(source)),
                bbox,
                itertools.repeat(bbox_latlon)
            )
        else:
            yield from files


def gen_zonal_stats(
        interface: Union[dict, GeoSeries, GeoDataFrame, str],
        raster: str,
        **kwargs
) -> np.ndarray:
    """
    :param interface: dict with 'features' key OR geopandas object OR shapely file path
    :param raster: .tif file path
    :param kwargs: kwargs to pass to rasterstats.gen_zonal_stats()
    :return: np.ndarray
    """
    stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
    rows = len(interface['features'])
    columns = len(stats)
    gen = rasterstats.gen_zonal_stats(
        interface,
        raster,
        stats=stats,
        # stats=stats,
        # nodata=0,
        # nodata=-1,
        **kwargs
    )

    # chain = itertools.chain.from_iterable(map(dict.values, gen))

    # # # I am assuming that the output dict keys match the order of stats however I am not sure I have this guarantee
    # def assertion() -> Iterator[Iterable[float]]:
    #     for output in gen:
    #         assert all(a == b for a, b in zip(output.keys(), stats))
    #         yield from output.values()
    #         # yield output.values()
    #
    # chain = assertion()
    #
    chain = itertools.chain.from_iterable(map(dict.values, gen))

    arr = np.fromiter(chain, dtype=np.float64, count=rows * columns)
    # arr /= 255
    arr = arr.reshape((rows, columns))
    return arr


class RasterStats(abc.ABC):
    def overlay( self, **kwargs):
        overlay(self.rasterstats, self.raster, **kwargs)

    @property
    def raster(self) -> str:
        return self._raster()

    @abc.abstractmethod
    def _raster(self) -> str:
        ...


    @classmethod
    @abc.abstractmethod
    def _rasterstats_from_file(
            cls,
            file: str,
            shadow_dir: str,
            zoom: int,
            threshold: tuple[float, float],
            mask: Optional[list[float]],
            raster: Optional[str],
    ) -> GeoDataFrame:
        ...

    @classmethod
    def rasterstats_from_file(
            cls,
            file: Union[str, list[str]],
            shadow_dir: str,
            zoom: int,
            threshold: tuple[float, float] = (0.0, 1.0),
            mask: Optional[list[float]] = None,
            raster: Optional[str] = None,
    ) -> GeoDataFrame:
        """
        :param file: .pbf file or pyrosm source
        :param shadow_dir: directory of all shadow xtiles and ytiles
        :param zoom: slippy map zoom
        :param threshold: cut-off threshold for array values
        :param mask: [miny, minx, maxy, maxx] to restrict raster array
        :param raster: optional raster file that will be used instead of generating a raster for the file
        :return: GeoDataFrame
            geometry as original lines, even though the area is a 4m buffer
        """
        return cls._rasterstats_from_file(
            file,
            shadow_dir=shadow_dir,
            zoom=zoom,
            threshold=threshold,
            mask=mask,
            raster=raster,
        )

    @abc.abstractmethod
    def _rasterstats(self) -> GeoDataFrame:
        ...

    @property
    def rasterstats(self, ) -> GeoDataFrame:
        return self._rasterstats()

    @classmethod
    @abc.abstractmethod
    def _rastersize_from_file(
            cls,
            file: Union[str, Iterable[str]],
            zoom: int,
            mask: Optional[list[float]]
    ) -> int:
        ...

    def rastersize(
            self,
            zoom: int,
            mask: Optional[list[float]] = None
    ) -> int:
        """

        :param zoom: slippy tile zoom
        :param mask: mask to restrict raster [miny, minx, maxy, maxx]
        :return: raster array size in bytes
        """
        return self._rastersize(zoom, mask)

    @abc.abstractmethod
    def _rastersize(self, zoom: int, mask: Optional[list[float]]):
        ...

    @classmethod
    def rastersize_from_file(
            cls,
            file: Union[str, Iterable[str]],
            zoom: int,
            mask: Optional[list[float]] = None,
    ) -> int:
        """
        :param file: osm.pbf file,
        :param zoom: slippy tile zoom
        :param mask: mask to restrict raster [miny, minx, maxy, maxx]
        :return: raster array size in bytes
        """
        return cls._rastersize_from_file(file, zoom=zoom, mask=mask)

    @classmethod
    def rastersize_from_files(
            cls,
            files: Iterable[str],
            zoom: int,
            mask: Optional[Iterable[list[float]]] = None
    ) -> dict[str, int]:
        """
        :param files: osm.pbf files
        :param zoom: slippy tile zoom
        :param mask: masks to restrict rasters [[miny, minx, maxy, maxx]]
        :return: {file: raster array size}
        """
        if mask is None:
            mask = itertools.repeat(None)
        return {
            f: cls._rastersize_from_file(f, zoom=zoom, mask=m)
            for f, m in zip(files, mask)
        }


class MetaRasterOsmium(type(osmium.SimpleHandler), type(abc.ABC), metaclass=type):
    """
    Metaclasses are typically too complicated to be necessary, but the polymorphism of inheriting from both
    osmium.SimpleHandler and RasterStats requires a metaclass that inherits from both of the respective metaclasses,
    Because nothing is implemented here, it inherits __new__ and __init__ directly from osmium.SimpleHandler
    """


class DescriptorParks(osmium.SimpleHandler, RasterStats, metaclass=MetaRasterOsmium):
    wkbfab = osmium.geom.WKBFactory()

    def _raster(self) -> str:
        surfaces = self._surfaces
        return get_raster_path(
            *self.rasterstats.total_bounds,
            basedir=surfaces.shadow_dir,
            threshold=surfaces.threshold,
            mask=surfaces.mask,
            zoom=surfaces.zoom
        )

    def _rastersize(self, zoom: int, mask: Optional[list[float]]):
        gdf = self.gdf
        return get_raster_size(*gdf.total_bounds, zoom=zoom, mask=mask)

    @classmethod
    def _rastersize_from_file(
            cls,
            file: Union[str, Iterable[str]],
            zoom: int,
            mask: Optional[list[float]],
    ) -> int:
        if '.' not in file:
            file = pyrosm.get_data(file)
        surfaces = Surfaces(file, shadow_dir='', zoom=zoom, mask=mask)
        gdf = surfaces.parks.gdf
        return get_raster_size(*gdf.total_bounds, zoom=zoom, mask=mask)

    def _rasterstats(self) -> GeoDataFrame:
        surfaces = self._surfaces
        if surfaces in self._rasterstats_:
            return self._rasterstats_[surfaces]

        gdf = self.gdf
        mask = surfaces.mask
        if mask is not None:
            box = shapely.geometry.box(mask[1], mask[0], mask[3], mask[2])
            gdf = gdf.clip(box)
        raster = get_raster_path(
            *gdf.total_bounds,
            zoom=surfaces.zoom,
            basedir=surfaces.shadow_dir,
            threshold=surfaces.threshold,
            outpath=surfaces.raster,
            mask=surfaces.mask,
        )

        # Some geometry can be None and rasterstats will raise an exception
        # loc = gdf['geometry'].notna()
        # geometry: GeoSeries = gdf.loc[loc, 'geometry']
        gdf = gdf[gdf['geometry'].notna()]
        geometry = gdf.geometry

        step = math.ceil(len(geometry) / multiprocessing.cpu_count())
        slices = [
            slice(left, left + step)
            for left in range(0, len(geometry), step)
        ]
        interfaces = [
            getattr(geometry.iloc[s], '__geo_interface__')
            for s in slices
        ]

        with concurrent.futures.ProcessPoolExecutor() as processes:
            results = processes.map(gen_zonal_stats, interfaces, itertools.repeat(raster))
            results = list(results)
        arr = np.concatenate(results)

        stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
        data = {
            stat: arr[:, i]
            for i, stat in enumerate(stats)
        }

        centroid: shapely.geometry.Point = geometry.iloc[0].centroid
        lon = centroid.x
        lat = centroid.y
        crs = get_utm_from_lon_lat(lon, lat)
        area = (
            geometry.to_crs(crs)
                .area
                .values
        )
        data['area'] = area
        data['name'] = gdf['name'].values

        result = GeoDataFrame(
            data=data,
            index=gdf.index,
            crs=gdf.crs,
            geometry=gdf.geometry,
        )
        stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
        normalize = set('min max mean sum median'.split())
        asint = {'nodata', 'count'}
        for stat in stats:
            if stat in normalize:
                result[stat] = result[stat] / 255
            elif stat in asint:
                result[stat] = result[stat].astype('Int64')

        result['name'] = Series.astype(result['name'], 'string')

        self._rasterstats_[surfaces] = result
        return result

    @classmethod
    def _rasterstats_from_file(
            cls,
            file: str,
            shadow_dir: str,
            zoom: int,
            threshold: tuple[float, float],
            mask: Optional[list[float]],
            raster: Optional[str],
    ) -> GeoDataFrame:
        if '.' not in file:
            file = pyrosm.get_data(file)
        surfaces = Surfaces(file, shadow_dir, zoom, threshold, mask, raster)
        parks = surfaces.parks
        return parks.rasterstats
        #
        # gdf = parks.gdf
        # raster = get_raster_path(
        #     *gdf.total_bounds,
        #     zoom=zoom,
        #     basedir=shadow_dir,
        #     threshold=threshold,
        #     outpath=raster,
        #     mask=mask,
        # )
        #
        # # Some geometry can be None and rasterstats will raise an exception
        # # loc = gdf['geometry'].notna()
        # # geometry: GeoSeries = gdf.loc[loc, 'geometry']
        # gdf = gdf[gdf['geometry'].notna()]
        # geometry = gdf.geometry
        #
        # step = math.ceil(len(geometry) / multiprocessing.cpu_count())
        # slices = [
        #     slice(l, l + step)
        #     for l in range(0, len(geometry), step)
        # ]
        # interfaces = [
        #     getattr(geometry.iloc[s], '__geo_interface__')
        #     for s in slices
        # ]
        #
        # with concurrent.futures.ProcessPoolExecutor() as processes:
        #     results = processes.map(gen_zonal_stats, interfaces, itertools.repeat(raster))
        #     results = list(results)
        # arr = np.concatenate(results)
        #
        # data = {
        #     stat: arr[:, i]
        #     for i, stat in enumerate(stats)
        # }
        #
        # centroid: shapely.geometry.Point = geometry.iloc[0].centroid
        # lon = centroid.x
        # lat = centroid.y
        # crs = get_utm_from_lon_lat(lon, lat)
        # area = (
        #     geometry.to_crs(crs)
        #         .area
        #         .values
        # )
        # data['area'] = area
        # data['name'] = gdf['name'].values
        #
        # result = GeoDataFrame(
        #     data=data,
        #     index=gdf.index,
        #     crs=gdf.crs,
        #     geometry=gdf.geometry,
        # )
        # stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
        # normalize = set('min max mean sum median'.split())
        # asint = {'nodata', 'count'}
        # for stat in stats:
        #     if stat in normalize:
        #         result[stat] = result[stat] / 255
        #     elif stat in asint:
        #         result[stat] = result[stat].astype('Int64')
        #
        # result['name'] = Series.astype(result['name'], 'string')
        # return result
        #

    def __init__(self):
        self._rasterstats_: WeakKeyDictionary[Surfaces, GeoDataFrame] = WeakKeyDictionary()
        super(DescriptorParks, self).__init__()
        self.natural = {'wood', 'grass'}
        self.land_use = {
            'wood', 'grass', 'forest', 'orchard', 'village_green',
            'vineyard', 'cemetery', 'meadow', 'village_green',
        }
        self.leisure = {
            'dog_park', 'park', 'playground', 'recreation_ground',
        }
        self.name = {}
        self.geometry = {}
        self.ways = set()
        self._cache: WeakKeyDictionary[Surfaces, GeoDataFrame] = WeakKeyDictionary()
        self._bbox: WeakKeyDictionary[Surfaces, list[float]] = WeakKeyDictionary()

    def area(self, a: osmium.osm.Area):
        # TODO: What about nodes marked 'point of interest'?
        tags: osmium.osm.TagList = a.tags
        # Qualifiers
        if not (
                tags.get('natural', None) in self.natural
                or tags.get('land_use', None) in self.land_use
                or tags.get('leisure', None) in self.leisure
        ):
            return

        id_ = a.orig_id()
        if a.from_way():
            self.ways.add(id_)

        try:
            self.geometry[id_] = self.wkbfab.create_multipolygon(a)
        except RuntimeError:
            ...
        if 'name' in tags:
            self.name[id_] = tags['name']

    def apply_file(self, filename, locations=False, idx='flex_mem'):
        for key, value in self.__dict__.items():
            if not key.startswith('_') and isinstance(value, dict):
                value.clear()
        super(DescriptorParks, self).apply_file(filename, locations, idx)

    def __get__(self, instance: 'Surfaces', owner: Type['Surfaces']) -> 'DescriptorParks':
        self._surfaces = instance
        return self

    def __set__(self, instance, value):
        self._cache[instance] = value

    def __delete__(self, instance):
        del self._cache[instance]

    @property
    def gdf(self) -> GeoDataFrame:
        surfaces = self._surfaces
        if surfaces not in self._cache:
            self.apply_file(surfaces.file, locations=True)
            index = np.fromiter(self.geometry.keys(), dtype=np.uint64, count=len(self.geometry))
            geometry = GeoSeries.from_wkb(list(self.geometry.values()), index=index)

            index = np.fromiter(self.name.keys(), dtype=np.uint64, count=len(self.name))
            name = np.fromiter(self.name.values(), dtype='U128', count=len(self.name))
            name = Series(name, index=index, dtype='string')

            index = np.fromiter(self.ways, dtype=np.uint64, count=len(self.ways))
            ways = np.full(len(self.ways), True, dtype=bool)
            ways = Series(ways, index=index, dtype='boolean')

            # if np.all(name.index.values == geometry.index.values):
            #     raise RuntimeError('you can do this better')
            gdf = GeoDataFrame({
                'name': name,
                'way': ways,
            }, crs=4326, geometry=geometry)
            gdf.loc[gdf['way'].isna(), 'way'] = False
            self.__set__(surfaces, gdf)
            return gdf
        return self._cache[surfaces]

    @gdf.setter
    def gdf(self, value):
        self._cache[self._surfaces] = value

    @gdf.deleter
    def gdf(self):
        del self._cache[self._surfaces]


class DescriptorNetwork(RasterStats):
    network_type: str

    def _raster(self) -> str:
        surfaces = self.networks.surfaces
        return get_raster_path(
            *self.rasterstats.total_bounds,
            basedir=surfaces.shadow_dir,
            threshold=surfaces.threshold,
            mask=surfaces.mask,
            zoom=surfaces.zoom,
        )

    def _rastersize(self, zoom: int, mask: Optional[list[float]]):
        gdf = self.gdf
        return get_raster_size(*gdf.total_bounds, zoom=zoom, mask=mask)

    @classmethod
    def _rastersize_from_file(
            cls,
            file: Union[str, Iterable[str]],
            zoom: int,
            mask: Optional[list[float]],
    ) -> Union[float, dict[str, float]]:
        if '.' not in file:
            file = pyrosm.get_data(file)
        surfaces = Surfaces(file, shadow_dir='', zoom=zoom, mask=mask)
        network: DescriptorNetwork = surfaces.networks.__getattribute__(cls.network_type)
        gdf = network.gdf
        return get_raster_size(*gdf.total_bounds, zoom=zoom, mask=mask)

    def _rasterstats(self) -> GeoDataFrame:
        surfaces = self.networks.surfaces
        if surfaces in self._rasterstats_:
            return self._rasterstats_[surfaces]

        gdf = self.gdf
        mask = surfaces.mask
        if mask is not None:
            box = shapely.geometry.box(mask[1], mask[0], mask[3], mask[2])
            gdf = gdf.clip(box)

        # Some geometry can be None and rasterstats will raise an exception
        # This was a significant slip-up. I should have just dropped the NA geometries from the entire GDF
        # loc = gdf.geometry.notna()
        # geometry: GeoSeries = gdf.loc[loc, 'geometry']
        gdf: GeoDataFrame = gdf[gdf.geometry.notna()]
        geometry = gdf.geometry

        # We are working with lines so we are buffering each line by 4 meters, which is about a car lane
        centroid: shapely.geometry.Point = gdf.geometry.iloc[0].centroid
        lon = centroid.x
        lat = centroid.y
        crs = get_utm_from_lon_lat(lon, lat)
        buffer = (geometry
                  .to_crs(crs)
                  .buffer(2)  # was originally a 4 meter buffer, but a lane is 4 meters wide so should be 2 meters
                  # probably doesn't matter too much
                  # .buffer(4)
                  )
        area = buffer.area
        buffer = buffer.to_crs(4326)

        raster = get_raster_path(
            *buffer.total_bounds,
            zoom=surfaces.zoom,
            basedir=surfaces.shadow_dir,
            threshold=surfaces.threshold,
            outpath=surfaces.raster,
            mask=surfaces.mask,
        )

        step = math.ceil(len(geometry) / multiprocessing.cpu_count())
        slices = [
            slice(left, left + step)
            for left in range(0, len(buffer), step)
        ]
        interfaces = [
            getattr(buffer[s], '__geo_interface__')
            for s in slices
        ]

        with concurrent.futures.ProcessPoolExecutor() as processes:
            results = processes.map(gen_zonal_stats, interfaces, itertools.repeat(raster))
            results = list(results)

        arr = np.concatenate(results)

        stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
        columns = {
            stat: arr[:, i]
            for i, stat in enumerate(stats)
        }

        columns['area'] = area
        columns['name'] = gdf['name']

        result = GeoDataFrame(
            columns,
            index=gdf.index,
            crs=gdf.crs,
            geometry=geometry.values
        )
        normalize = set('min max mean sum median'.split())
        asint = {'nodata', 'count'}
        for stat in stats:
            if stat in normalize:
                result[stat] = result[stat] / 255
            elif stat in asint:
                result[stat] = result[stat].astype('Int64')

        result['name'] = Series.astype(result['name'], 'string')

        self._rasterstats_[surfaces] = result
        return result

    @classmethod
    def _rasterstats_from_file(
            cls,
            file: str,
            shadow_dir: str,
            zoom: int,
            threshold: tuple[float, float],
            mask: Optional[list[float]],
            raster: Optional[str],
    ) -> GeoDataFrame:
        if '.' not in file:
            file = pyrosm.get_data(file)
        surfaces = Surfaces(file, shadow_dir, zoom, threshold, mask, raster)
        network: DescriptorNetwork = surfaces.networks.__getattribute__(cls.network_type)
        return network.rasterstats
        # network: DescriptorNetwork = surfaces.networks.__getattribute__(cls.network_type)

        # return network.rasterstats(zoom, threshold, mask, raster)
        # # gdf = network.gdf
        #
        # # Some geometry can be None and rasterstats will raise an exception
        # # This was a significant slip-up. I should have just dropped the NA geometries from the entire GDF
        # # loc = gdf.geometry.notna()
        # # geometry: GeoSeries = gdf.loc[loc, 'geometry']
        # gdf: GeoDataFrame = gdf[gdf.geometry.notna()]
        # geometry = gdf.geometry
        #
        # # We are working with lines so we are buffering each line by 4 meters, which is about a car lane
        # centroid: shapely.geometry.Point = gdf.geometry.iloc[0].centroid
        # lon = centroid.x
        # lat = centroid.y
        # crs = get_utm_from_lon_lat(lon, lat)
        # buffer = (
        #     geometry.to_crs(crs)
        #         .buffer(2)  # was originally a 4 meter buffer, but a lane is 4 meters wide so should be 2 meters
        #     # probably doesn't matter too much
        #     # .buffer(4)
        # )
        # area = buffer.area
        # buffer = buffer.to_crs(4326)
        #
        # raster = get_raster_path(
        #     *buffer.total_bounds,
        #     zoom=zoom,
        #     basedir=shadow_dir,
        #     threshold=threshold,
        #     outpath=raster,
        #     mask=mask,
        # )
        #
        # step = math.ceil(len(geometry) / multiprocessing.cpu_count())
        # slices = [
        #     slice(l, l + step)
        #     for l in range(0, len(buffer), step)
        # ]
        # interfaces = [
        #     getattr(buffer[s], '__geo_interface__')
        #     for s in slices
        # ]
        #
        # with concurrent.futures.ProcessPoolExecutor() as processes:
        #     results = processes.map(gen_zonal_stats, interfaces, itertools.repeat(raster))
        #     results = list(results)
        #
        # arr = np.concatenate(results)
        #
        # stats: Collection[str] = tuple('min max mean count sum median nodata'.split())
        # columns = {
        #     stat: arr[:, i]
        #     for i, stat in enumerate(stats)
        # }
        #
        # columns['area'] = area
        # columns['name'] = gdf['name']
        #
        # result = GeoDataFrame(
        #     columns,
        #     index=gdf.index,
        #     crs=gdf.crs,
        #     geometry=geometry.values
        # )
        # normalize = set('min max mean sum median'.split())
        # asint = {'nodata', 'count'}
        # for stat in stats:
        #     if stat in normalize:
        #         result[stat] = result[stat] / 255
        #     elif stat in asint:
        #         result[stat] = result[stat].astype('Int64')
        #
        # result['name'] = Series.astype(result['name'], 'string')
        # return result
        #

    def __get__(self, instance: 'DescriptorNetworks', owner: Type['DescriptorNetworks']):
        self.networks = instance
        return self

    def _get_network(self) -> tuple[GeoDataFrame, GeoDataFrame]:
        networks = self.networks
        surfaces = self.networks.surfaces
        osm: pyrosm.OSM = networks._osm[surfaces]
        # nodes, geometry = osm.get_network(self.network_type, None, True)
        nodes, geometry = None, osm.get_network(self.network_type, None, False)
        self._bbox[surfaces] = geometry.total_bounds

        nodes: Optional[GeoDataFrame]
        geometry: GeoDataFrame
        if 'u' in geometry:
            geometry = geometry['id name geometry u v length surface tunnel'.split()]
        else:
            geometry = geometry['id name geometry length tunnel'.split()]

        loc = pd.Series.isin(geometry['tunnel'], {'passage', 'yes', 'building_passage', 'covered'})
        geometry = geometry.loc[~loc]
        geometry = geometry.drop('tunnel', axis=1)
        geometry = geometry.set_index('id')
        return nodes, geometry

    def __init__(self):
        self._cache: WeakKeyDictionary[Surfaces, tuple[GeoDataFrame, GeoDataFrame]] = WeakKeyDictionary()
        self._bbox: WeakKeyDictionary[Surfaces, list[float]] = WeakKeyDictionary()
        self._rasterstats_: WeakKeyDictionary[Surfaces, GeoDataFrame] = WeakKeyDictionary()

    @property
    def gdf(self) -> GeoDataFrame:
        surfaces = self.networks.surfaces
        if surfaces not in self._cache:
            self._cache[surfaces] = self._get_network()
        return self._cache[surfaces][1]
        # if instance not in self._cache:
        #     self._cache[instance] = self._get_network()
        # return self._cache[instance][1]

    @gdf.setter
    def gdf(self, value):
        surfaces = self.networks.surfaces
        nodes, geometry = self._cache[surfaces]
        self._cache[surfaces] = (nodes, value)

    @gdf.deleter
    def gdf(self):
        del self._cache[self.networks.surfaces]

    @property
    def nodes(self) -> GeoDataFrame:
        surfaces = self.networks.surfaces
        if surfaces not in self._cache:
            self._cache[surfaces] = self._get_network()
        return self._cache[surfaces][0]

    @nodes.setter
    def nodes(self, value):
        surfaces = self.networks.surfaces
        nodes, geometry = self._cache[surfaces]
        self._cache[surfaces] = (value, geometry)

    @nodes.deleter
    def nodes(self):
        del self._cache[self.networks.surfaces]


class DescriptorWalkingNetwork(DescriptorNetwork):
    network_type = 'walking'


class DescriptorCyclingNetwork(DescriptorNetwork):
    network_type = 'cycling'


class DescriptorDrivingNetwork(DescriptorNetwork):
    network_type = 'driving'


class DescriptorDrivingServiceNetwork(DescriptorNetwork):
    network_type = 'driving+service'


class DescriptorAllNetwork(DescriptorNetwork):
    network_type = 'all'


class DescriptorNetworks:
    walking = DescriptorWalkingNetwork()
    cycling = DescriptorCyclingNetwork()
    driving = DescriptorDrivingNetwork()
    driving_service = DescriptorDrivingServiceNetwork()
    all = DescriptorAllNetwork()

    def __init__(self):
        self.surfaces: Optional['Surfaces'] = None
        self._osm: WeakKeyDictionary[Surfaces, pyrosm.OSM] = WeakKeyDictionary()

    def __get__(self, instance: 'Surfaces', owner):
        self.surfaces = instance
        if instance is not None and instance not in self._osm:
            self._osm[instance] = pyrosm.OSM(self.surfaces.file)
        return self


class Surfaces:
    parks = DescriptorParks()
    networks = DescriptorNetworks()


    def __init__(
            self,
            file: str,
            shadow_dir: str,
            zoom: int,
            threshold: tuple[float, float] = (0.0, 1.0),
            mask: Optional[list[float]] = None,
            raster: Optional[str] = None,
    ):
        if file.rpartition('.')[2] != 'pbf':
            raise ValueError(f"{file=} is not a PBF file")
        self.file = file
        self.shadow_dir = shadow_dir
        self.zoom = zoom
        self.threshold = threshold
        self.mask = mask
        self.raster = raster

    @classmethod
    def concatenate_from_files(cls, files: list[str]) -> DataFrame:
        """
        :param files: list of .feather files that will be concatenated for comparison across cities
        :return     a single Dataframe, with the OSM ID and filename as index, and statistics as values
        """
        names = (
            file.rpartition('.')[0]
            for file in files
        )

        def gdfs() -> Iterator[GeoDataFrame]:
            it_files = iter(files)
            with concurrent.futures.ThreadPoolExecutor() as threads:
                prev_future = threads.submit(gpd.read_feather, next(it_files))
                try:
                    next_future = threads.submit(gpd.read_feather, next(it_files))
                except StopIteration:
                    yield prev_future.result()
                    return
                for file in files:
                    yield prev_future.result()
                    prev_future = next_future
                    next_future = threads.submit(gpd.read_feather, file)
                yield next_future.result()

        concat = pd.concat((
            gdf.drop('geometry', axis=1)
                .assign(name=next(names))
                .set_index('name', append=True)
            for gdf in gdfs()
        ))
        return concat


#
if __name__ == '__main__':
    mask_ = [41.85784676139911, -87.64350384884648, 41.88852432376579, -87.61311978580892]
    loop_path = osmium_extract(
        '/home/arstneio/Downloads/chi.osm.pbf',
        '~/PycharmProjects/StaticOSM/work/osmium-tool/build/osmium',
        [41.85784676139911, -87.64350384884648, 41.88852432376579, -87.61311978580892]
    )
    driving = Surfaces.networks.driving.rastersize_from_file(
        loop_path,
        16,
        mask=mask_
    )
    parks = Surfaces.parks.rastersize_from_file(
        loop_path,
        16,
        mask=mask_
    )
    print()

    # network = Surfaces.networks.driving.rasterstats_from_file(
    #     loop_path,
    #     '/home/arstneio/Downloads/shadows_new/chi-summer',
    #     16,
    # )
    # print(f'{(time.time() - t) / 60:.1f} minutes')
    # network.to_feather('chicago.feather')
    # print()
    #

# if __name__ == '__main__':
#     print('surfaces')
#     chicago = pyrosm_extract(
#         'chicago',
#         osmium_executable_path='~/PycharmProjects/StaticOSM/work/osmium-tool/build/osmium',
#         bbox=[41.865140845410046, -87.634181491039, 41.88789218539278, -87.61083554343192],
#     )
#     sao_res = Surfaces.networks.driving.rasterstats_from_file(
#         chicago,
#             '/home/arstneio/Downloads/shadows/test/winter/',
#         zoom=16,
#     )
#     print(sao_res.total_bounds)
#
#     london = pyrosm_extract(
#         'london',
#         osmium_executable_path='~/PycharmProjects/StaticOSM/work/osmium-tool/build/osmium',
#         bbox=[51.48810230578064, -0.02620147457317379, 51.50680415101511, 0.0001485471745494128],
#     )
#     l_res = Surfaces.networks.driving.rasterstats_from_file(
#         london,
#         '/home/arstneio/Downloads/shadows/test/winter/',
#         zoom=16,
#     )
#     print(l_res.total_bounds)
#     print()

# t = time.time()
# driving = Surfaces.networks.driving.rasterstats_from_file(
#     'london',
#     '/home/arstneio/Downloads/shadows/test/winter/',
#     zoom=16,
#     # threshold=.25
# )

# print(f'parks took {int(time.time() - t)} seconds; {len(parks)=}')
# parks = Surfaces.parks.rasterstats_from_file(
#     '/home/arstneio/Downloads/ams.osm.pbf',
#     '/home/arstneio/Downloads/shadows/test/winter/',
#     zoom=16,
#     # threshold=.25
# )
# print()
# t = time.time()
# networks = Surfaces.networks.driving.rasterstats_from_file(
#     path,
#     '/home/arstneio/Downloads/shadows/test/winter/',
#     zoom=16,
#     # threshold=.25
# )
# print(f'driving networks took {int(time.time() - t)} seconds; {len(networks)=}')
# print()
