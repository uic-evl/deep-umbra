import concurrent.futures
import itertools
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm.notebook import tqdm

from deep_shadow import *
from utils import *

deep_shadow = DeepShadow(512, 512)
deep_shadow.restore('training_checkpoints/evaluation_new/uniform_cities')


class Cache(UserDict):
    def __setitem__(self, key, value):
        value = tf.io.decode_png(value)[:, :, 0]
        value = tf.reshape(value, (256, 256, 1))
        value = tf.cast(value, tf.float32)
        super().__setitem__(key, value)


def load_input(height_path, city, zoom, i, j):
    filename = tf.strings.format('{}/{}/{}/{}/{}.png', (height_path, city, zoom, i, j))
    filename = tf.strings.regex_replace(filename, '\"', "")
    input_image = tf.io.read_file(filename)
    return input_image


def predict(
        height_folder: Path,
        output_folder: Path,
        dates: tuple[str] = ('spring', 'summer', 'winter'),
):
    height_folder = Path(height_folder)
    output_folder = Path(output_folder)

    paths = height_folder.glob('*/')
    filepaths = list(itertools.chain.from_iterable(
        map(lambda path: Path.glob(path, '*/*/*.png'), paths)
    ))
    city = (filepaths[0].parts[-4])
    zoom = int(filepaths[0].parts[-3])
    ij = set(zip(
        map(lambda path: int(Path.parts[-2]), filepaths),
        map(lambda path: int(Path.parts[-1].rpartition('.')[0]), filepaths)
    ))
    W = min(i for i, _ in ij)
    E = max(i for i, _ in ij)
    N = min(j for _, j in ij)
    S = max(j for _, j in ij)

    xy_indices = {
        (x, y): [
            (xx, yy)
            for xx in range(256 + 256 * x, 256 + 256 * (x + 1))
            for yy in range(256 + 256 * y, 256 + 256 * (y + 1))
        ]
        for x, y in itertools.product(range(-1, 2), range(-1, 2))
    }

    list_xy = list(itertools.product(
        (-1, 0, 1),
        (-1, 0, 1),
    ))
    futures: dict[tuple[int, int], concurrent.futures.Future] = {}
    cache: dict[tuple[int, int], np.ndarray] = Cache()
    serialize = []
    threads = ThreadPoolExecutor()
    E_MINUS1 = E - 1
    W_PLUS1 = W + 1

    # PREPARE NW CORNER
    for pair in itertools.product((W, W + 1), (N, N + 1)):
        if pair not in ij:
            continue
        futures[(W, N)] = threads.submit(load_input, height_folder, city, zoom, *pair)

    for i, j in tqdm(itertools.product(
            range(W, E + 1),
            range(N, S + 1),
    )):
        # PREPARE
        if i < E_MINUS1:
            # NEXT: EASTWARD
            se = (i + 2, j + 1)
            if se in ij:
                futures[se] = threads.submit(load_input, height_folder, city, zoom, *se)
        elif i == E:
            # NEXT: WEST EDGE
            s = (W, j + 2)
            if s in ij:
                futures[s] = threads.submit(load_input, height_folder, city, zoom, *s)

            se = (W + 1, j + 2)
            if se in ij:
                futures[se] = threads.submit(load_input, height_folder, city, zoom, *se)

        # LOAD
        if i < E:
            # NEXT: EASTWARD
            se = (i + 1, j + 1)
            if se in futures:
                cache[se] = futures[se].result()
                del futures[se]
        if i == W:
            # WEST EDGE
            s = (i, j + 1)
            if s in futures:
                cache[s] = futures[s].result()
                del futures[s]

        # DELETE
        if i == W:
            # WEST EDGE
            ne = (E, j - 2)
            if ne in cache:
                del cache[ne]
            n = (E - 1, j - 2)
            if n in cache:
                del cache[n]
        elif W_PLUS1 < i < E_MINUS1:
            # EASTWARD
            nw = (i - 2, j - 2)
            if nw in cache:
                del cache[nw]

        # SKIP?
        if (i, j) not in cache:
            continue

        # LOAD INPUT GRID
        tf_height = tf.zeros((256 * 3, 256 * 3, 1))
        for xy in list_xy:
            x, y = xy
            pair = (i + x, j + y)
            if pair not in cache:
                continue
            piece = cache[pair]
            indices = xy_indices[xy]
            tf_height = tf.tensor_scatter_nd_update(tf_height, indices, piece)

        tf_height = tf_height[128:-128, 128:-128]
        tf_lat = tf.ones((512, 512,), dtype=tf.float32)
        lat, _ = num2deg(i, j, zoom)
        tf_lat = tf.math.scalar_mul(float(lat), tf_lat)
        tf_lat = tf.reshape(tf_lat, (512, 512, 1))

        tf_date = tf.ones((512, 512,), dtype=tf.float32)

        for date in dates:
            path: Path = output_folder / f'{city}-{date}' / zoom / i / j
            path.mkdir(parents=True, exist_ok=True)

            if date == 'winter':
                value = 0
            elif date == 'spring' or date == 'fall':
                value = 1
            else:
                value = 2

            tf_date = tf.math.scalar_mul(float(value), tf_date)
            tf_date = tf.reshape(tf_date, (512, 512, 1))
            tf_height, tf_lat, tf_date = normalize_input(tf_height, tf_lat, tf_date)

            tf_height = np.array(tf_height).reshape(1, 512, 512, 1)
            tf_lat = np.array(tf_lat).reshape(1, 512, 512, 1)
            tf_date = np.array(tf_date).reshape(1, 512, 512, 1)

            prediction = deep_shadow.generator((tf_height, tf_lat, tf_date), training=True)

            prediction = prediction.numpy()[:, 128:-128, 128:-128, :]
            prediction = prediction * .5 + .5

            serialize.append(threads.submit(cv2.imwrite, path, prediction[0] * 255))

    # check for exceptions, can be commented
    for future in serialize:
        if future.exception():
            raise future.exception()

if __name__ == '__main__':
    height_folder = Path('data/heights_new')
    output_folder = Path('data/shadows_new')
    predict(height_folder, output_folder)
