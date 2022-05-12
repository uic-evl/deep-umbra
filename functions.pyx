import cv2
import cython
import numpy as np
cimport numpy as np
cimport libc.math as math

ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
if False | False:
    import math

# TODO: Perhaps specify the dtype in load_image() for better/worse quality
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef _load_image(
        np.ndarray[UINT64, ndim=1] cn,
        np.ndarray[UINT64, ndim=1] cw,
        np.ndarray[UINT16, ndim=1] weights,
        unsigned int cell_length,
):
    cdef np.ndarray[UINT16, ndim=2] grid = np.zeros((cell_length, cell_length), dtype=np.uint16)
    cdef unsigned int length = len(cw)
    cdef unsigned int k
    for k in range(length):
        grid[cn[k], cw[k]] = weights[k]
    return cv2.resize(grid, dsize=(256, 256))

def load_image(
        cn: np.ndarray,
        cw: np.ndarray,
        weights: np.ndarray,
        cell_length: int,
) -> np.ndarray:
    """

    :param cn:
    :param cw:
    :param weights:
    :param cell_length:
    :param absolute_max:
    :return: 256x256 image
    """
    result = _load_image(cn, cw, weights, cell_length)
    return result

cdef _deg2num(
        double lon_deg,
        double lat_deg,
        unsigned int zoom,
        bint always_xy
):
    cdef unsigned int n, xtile, ytile
    cdef double lat_rad
    # lat_rad = math.radians(lat_deg)
    lat_rad = lat_deg * math.pi / 180.0

    n = 2 ** zoom
    xtile = <unsigned int> ((lon_deg + 180) / 360 * n)
    ytile = <unsigned int> ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    if always_xy:
        return xtile, ytile
    else:
        return ytile, xtile

def deg2num(
        lon_deg: float,
        lat_deg: float,
        zoom: int,
        always_xy = True,
) -> tuple[int, int]:
    """

    :param lat_deg:
    :param lon_deg:
    :param zoom:
    :return: xtile, ytile
    """
    return _deg2num(lon_deg, lat_deg, zoom, always_xy)

cdef _num2deg(
        double xtile,
        double ytile,
        unsigned int zoom,
        bint always_xy
):
    cdef unsigned int n
    cdef double lon_deg, lat_rad, lat_deg
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

def num2deg(xtile: float, ytile: float, zoom: int, always_xy = False) -> tuple[float, float]:
    """

    :param xtile:
    :param ytile:
    :param zoom:
    :return: latitude, longitude
    """
    return _num2deg(xtile, ytile, zoom, always_xy)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _degs2nums(
        np.ndarray[F64, ndim=1] lon_degs,
        np.ndarray[F64, ndim=1] lat_degs,
        unsigned char zoom,
        bint always_xy,
):
    cdef unsigned int length = len(lat_degs)
    cdef np.ndarray[UINT, ndim = 1] xtiles = np.zeros((length), dtype=np.uint32)
    cdef np.ndarray[UINT, ndim = 1] ytiles = np.zeros((length), dtype=np.uint32)
    cdef unsigned int n = 2 ** zoom
    cdef unsigned int k
    for k in range(length):
        lat_rad = lat_degs[k] * math.pi / 180.0
        xtile = <unsigned int> ((lon_degs[k] + 180) / 360 * n)
        ytile = <unsigned int> ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        xtiles[k] = xtile
        ytiles[k] = ytile
    if always_xy:
        return xtiles, ytiles
    else:
        return ytiles, xtiles

def degs2nums(lon_degs: np.ndarray, lat_degs: np.ndarray, zoom: int, always_xy = True):
    """
    :param lat_degs:
    :param lon_degs:
    :param zoom:
    :return: xtile, ytile
    """
    return _degs2nums(lon_degs, lat_degs, zoom, always_xy)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _nums2degs(
        np.ndarray[UINT, ndim=1] xtiles,
        np.ndarray[UINT, ndim=1] ytiles,
        unsigned int zoom,
        bint always_xy,
):
    cdef unsigned int length = len(xtiles)
    cdef np.ndarray[F64, ndim = 1] lat_degs = np.zeros((length), dtype=np.float64)
    cdef np.ndarray[F64, ndim = 1] lon_degs = np.zeros((length), dtype=np.float64)
    cdef unsigned int n = 2 ** zoom
    cdef unsigned int k
    for k in range(length):
        lon_deg = 360.0 * xtiles[k] / n - 180
        lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytiles[k] / n)))
        lat_deg = lat_rad * 180.0 / math.pi
        lat_degs[k] = lat_deg
        lon_degs[k] = lon_deg
    if always_xy:
        return lon_degs, lat_degs
    else:
        return lat_degs, lon_degs

def nums2degs(
        xtiles: np.ndarray,
        ytiles: np.ndarray,
        zoom: int,
        always_xy = False
) -> tuple[np.ndarray, np.ndarray]:
    """

    :param xtiles:
    :param ytiles:
    :param zoom:
    :return: latitude, longitude
    """
    return _nums2degs(xtiles, ytiles, zoom, always_xy)