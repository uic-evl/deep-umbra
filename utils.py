import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import random
import skfmm
import os
from IPython import display
from sklearn.model_selection import train_test_split


def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def num2deg(xtile, ytile, zoom):
    n = tf.math.pow(2, zoom)
    lon_deg = float(xtile) / float(n) * 360.0 - 180.0
    lat_rad = tf.math.atan(tf.math.sinh(
        3.14159265359 * (1.0 - 2.0 * float(ytile) / float(n))))
    lat_deg = rad2deg(lat_rad)
    return (lat_deg, lon_deg)


def generate_images(model, input_height, input_lat, input_date, target, path, latitude=False, date=False, save=False):
    ip = [input_height]
    if latitude:
        ip.append(input_lat)
    if date:
        ip.append(input_date)

    prediction = model(ip, training=True)
    plot_comparison(input_height[0], target[0], prediction[0], path, save=save)


def plot_result(input_height, prediction):
    plt.figure(figsize=(15, 15))

    display_list = [input_height, prediction]
    title = ['Input Image', 'Predicted Image']

    plt.subplot(1, 3, 1)
    plt.title(title[0])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow((display_list[0] * 0.5 + 0.5))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(title[1])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow((display_list[1] * 0.5 + 0.5))
    plt.axis('off')

    plt.show()


def plot_comparison(input_height, target, prediction, path=None, save=False):
    plt.figure(figsize=(15, 6))

    target = target * 0.5 + 0.5
    prediction = prediction * 0.5 + 0.5

    mae = np.mean(np.abs(target-prediction))
    mse = np.mean((prediction - target) ** 2)
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    ssim = tf.image.ssim(prediction, target, max_val=1.0).numpy()

    # phi = np.where(target == 0, -1, 0) + 0.5

    # w = skfmm.distance(phi, dx = 1)
    # w = np. where(w < 0, 0, w)
    # # w = w / np.max(w) # normalize between 0 and 1

    # diff = tf.square(target - prediction)
    # weighted_diff = tf.multiply(w, diff)
    # weighted_rmse = tf.sqrt(tf.reduce_mean(weighted_diff))

    # print("mae: %f mse: %f rmse: %f ssim: %f" % (mae, mse, rmse, ssim))

    display_list = [input_height, target, prediction]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    plt.subplot(1, 3, 1)
    plt.title(title[0])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow((display_list[0] * 0.5 + 0.5))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(title[1])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow((display_list[1] * 0.5 + 0.5))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(title[2])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[2] * 0.5 + 0.5)
    plt.axis('off')

    if save:
        type = path.split('/')[-2]
        img_no = path.split('/')[-1]
        plt.suptitle("%s %s\n\nmae: %f mse: %f rmse: %f ssim: %f" %
                     (type, img_no, mae, mse, rmse, ssim))

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_input(height_path, city, zoom, i, j):

    # Read and decode an image file to a uint8 tensor
    filename = tf.strings.format(
        '{}/{}/{}/{}/{}.png', (height_path, city, zoom, i, j))
    filename = tf.strings.regex_replace(filename, '\"', "")
    input_image = tf.io.read_file(filename)
    input_image = tf.io.decode_png(input_image)[:, :, 0]

    input_image = tf.reshape(input_image, (256, 256, 1))
    input_image = tf.cast(input_image, tf.float32)

    return input_image


def load(height_path, shadow_path, city, date, zoom, i, j):

    filename = tf.strings.format(
        '{}/{}/{}/{}/{}.png', (height_path, city, zoom, i, j))
    filename = tf.strings.regex_replace(filename, '\"', "")
    input_image = tf.io.read_file(filename)
    input_image = tf.io.decode_png(input_image)[:, :, 0]

    filename = tf.strings.format(
        '{}/{}-{}/{}/{}/{}.png', (shadow_path, city, date, zoom, i, j))
    filename = tf.strings.regex_replace(filename, '\"', "")
    real_image = tf.io.read_file(filename)
    real_image = tf.io.decode_png(real_image)[:, :, 0]
    real_image = tf.experimental.numpy.where(input_image <= 0, real_image, 0)

    # new: loading the street tile images
    filename = tf.strings.format(
        './data/street_tile/{}/{}/{}/{}.png', (city, zoom, i, j))
    filename = tf.strings.regex_replace(filename, '\"', "")
    street_image = tf.io.read_file(filename)
    street_image = tf.io.decode_png(street_image)[:, :, 0]

    input_image = tf.reshape(input_image, (256, 256, 1))
    real_image = tf.reshape(real_image, (256, 256, 1))
    street_image = tf.reshape(street_image, (256, 256, 1))  # new

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    street_image = tf.cast(street_image, tf.float32)  # new

    return input_image, real_image, street_image  # new


def load_input_grid(height_path, city, date, zoom, i, j):
    all_input = tf.zeros((256*3, 256*3, 1))

    for x in range(-1, 2):
        for y in range(-1, 2):
            filepath = '%s/%s/%d/%d/%d.png' % (height_path,
                                               city, zoom, i+y, j+x)
            if os.path.isfile(filepath):
                iinput = load_input(height_path, city, zoom, i+y, j+x)
                indices = [(xx, yy) for xx in range(256+256*x, 256+256*(x+1))
                           for yy in range(256+256*y, 256+256*(y+1))]
                indices = np.array(indices).reshape(256, 256, -1)
                all_input = tf.tensor_scatter_nd_update(
                    all_input, indices, iinput)

    (latitude, longitude) = num2deg(i, j, zoom)

    all_input = all_input[128:-128, 128:-128]
    all_lat = tf.ones((512, 512), dtype=tf.float32)
    all_lat = tf.math.scalar_mul(float(latitude), all_lat)
    all_lat = tf.reshape(all_lat, (512, 512, 1))

    if date == 'winter':
        value = 0
    elif date == 'spring' or date == 'fall':
        value = 1
    else:
        value = 2

    all_date = tf.ones((512, 512), dtype=tf.float32)
    all_date = tf.math.scalar_mul(float(value), all_date)
    all_date = tf.reshape(all_date, (512, 512, 1))

    return all_input, all_lat, all_date


def load_grid(height_path, shadow_path, city, date, zoom, i, j, neighbors):
    all_input = tf.zeros((256*3, 256*3, 1))
    all_real = tf.zeros((256*3, 256*3, 1))
    all_street = tf.zeros((256*3, 256*3, 1))  # new

    count = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            #             if check_image(height_path, city, date, zoom, i+y, j+x):
            # tensorflow does not support slicing
            #                 all_input[256+256*x:256+256*(x+1),256+256*y:256+256*(y+1)], \
            #                 all_real[256+256*x:256+256*(x+1),256+256*y:256+256*(y+1)] \
            #                 = load(height_path, shadow_path, city, date, zoom, i+y, j+x)
            if neighbors[count] == 'True':
                iinput, real, street = load(height_path, shadow_path,
                                            city, date, zoom, i+y, j+x)  # new
                indices = [(xx, yy) for xx in range(256+256*x, 256+256*(x+1))
                           for yy in range(256+256*y, 256+256*(y+1))]
                indices = np.array(indices).reshape(256, 256, -1)
                all_input = tf.tensor_scatter_nd_update(
                    all_input, indices, iinput)
                all_real = tf.tensor_scatter_nd_update(all_real, indices, real)
                all_street = tf.tensor_scatter_nd_update(
                    all_street, indices, street)  # new

            count += 1

    (latitude, longitude) = num2deg(i, j, zoom)

    all_input = all_input[128:-128, 128:-128]
    all_real = all_real[128:-128, 128:-128]
    all_street = all_street[128:-128, 128:-128]  # new

    all_lat = tf.ones((512, 512), dtype=tf.float32)
    all_lat = tf.math.scalar_mul(float(latitude), all_lat)
    all_lat = tf.reshape(all_lat, (512, 512, 1))

    # solve tensorflow's unhashable problem
    if date == 'winter':
        value = 0
    elif date == 'spring' or date == 'fall':
        value = 1
    else:
        value = 2

    all_date = tf.ones((512, 512), dtype=tf.float32)
    all_date = tf.math.scalar_mul(float(value), all_date)
    all_date = tf.reshape(all_date, (512, 512, 1))

    return all_input, all_real, all_street, all_lat, all_date  # new


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, size):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, size, size, 1])

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]


def normalize_input(input_image, lat_image, date_image):
    input_image = (input_image / 127.5) - 1
    lat_image = ((lat_image + 90) / 90.0) - 1
    date_image = date_image - 1

    return input_image, lat_image, date_image


def normalize(input_image, real_image, street_image, lat_image, date_image):  # new
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    street_image = (street_image / 127.5) - 1  # new
    lat_image = ((lat_image + 90) / 90.0) - 1
    date_image = date_image - 1

    return input_image, real_image, street_image, lat_image, date_image  # new


@tf.function()
def random_jitter(input_image, real_image, lat_image, date_image):
    original_size = input_image.shape[0]
    new_size = int(original_size*1.10)
    # Resizing to 286x286
    input_image, real_image = resize(
        input_image, real_image, new_size, new_size)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(
        input_image, real_image, original_size)

#     if tf.random.uniform(()) > 0.5:
#         Random mirroring
#         input_image = tf.image.flip_left_right(input_image)
#         real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image, lat_image, date_image


def load_image(path):
    height_path, shadow_path, city, date, filename, neighbors = path[
        0], path[1], path[2], path[3], path[4], path[5:]

    filename = tf.strings.regex_replace(filename, '\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]
                                   ), int(tf.strings.split(tks[-1], '.')[0])

    input_image, real_image, street_image, lat_image, date_image = load_grid(
        height_path, shadow_path, city, date, zoom, i, j, neighbors)  # new
    input_image, real_image, street_image, lat_image, date_image = normalize(
        input_image, real_image, street_image, lat_image, date_image)  # new

    return input_image, real_image, street_image, lat_image, date_image, filename


def get_tiles(height_path, shadow_path, cities, dates, zoom, tiles_per_city):
    def get_path(row, city, date):
        values = [height_path, shadow_path, city, date,
                  '%d/%d/%d.png' % (row['zoom'], row['i'], row['j'])]
        for i in range(0, 9):
            values.append(str(row[str(i)]))
        return values

    all_dataset = []
    for city in cities:
        for date in dates:
            df = pd.read_csv('data/evaluation/%s-%s-%d.csv' %
                             (city, date, zoom))
            df = df.sample(frac=1, random_state=42).reset_index()  # shuffle

            # new: sample based on df['height']. sample tiles_per_city but 50% from df['height] <= 6.470588, 50% from df['height'] > 6.470588
            df1 = df[df['height'] <= 6.470588]
            df2 = df[df['height'] > 6.470588]
            df1 = df1.sample(
                n=min(tiles_per_city//2, len(df1)), random_state=42)
            df2 = df2.sample(
                n=min(tiles_per_city//2, len(df2)), random_state=42)
            df = pd.concat([df1, df2])

            # df = df.sample(n=min(tiles_per_city, len(df)), random_state=42)
            ds = df.apply(get_path, args=(city, date), axis=1).tolist()
            all_dataset.extend(ds)

    return all_dataset


def train_to_tensor(train_dataset, batch_size):

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset


def test_to_tensor(test_dataset, batch_size):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    test_dataset = test_dataset.map(load_image)
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset


def get_train_test(height_path, shadow_path, cities, dates, zoom, tiles_per_city, batch_size=2, train_size=0.6, ignore_images=[]):

    all_dataset = get_tiles(height_path, shadow_path,
                            cities, dates, zoom, tiles_per_city)

    if ignore_images:
        all_dataset = [x for x in all_dataset if x[4] not in ignore_images]

    train_dataset, test_dataset = train_test_split(
        all_dataset, train_size=train_size, random_state=42)

    train_dataset = train_to_tensor(train_dataset, batch_size)
    test_dataset = test_to_tensor(test_dataset, batch_size)

    return train_dataset, test_dataset


def get_metrics(test_dataset, generator, latitude=False, date=False):

    def sobel(img): return tf.image.sobel_edges(img)
    rmses = []
    maes = []
    mses = []
    ssims = []
    sobels = []

    for test_input, test_target, test_street, test_latitude, test_date, _ in test_dataset:
        ip = [test_input]
        if latitude:
            ip.append(test_latitude)
        if date:
            ip.append(test_date)

        prediction = generator(ip, training=True)
        # We did not do this before. This is to remove shadows from buildings in the prediction
        prediction = np.where(test_input > -1, -1, prediction)

        target = test_target.numpy()[:, 128:-128, 128:-128, :]
        prediction = prediction[:, 128:-128, 128:-128, :]
        test_street = test_street.numpy()[:, 128:-128, 128:-128, :]  # new

        target = target * 0.5 + 0.5
        prediction = prediction * 0.5 + 0.5
        test_street = test_street * 0.5 + 0.5  # new

        # new: consider only shadows on streets for both target and prediction
        target = np.where(test_street > 0, target, 0)
        prediction = np.where(test_street > 0, prediction, 0)

        # new
        if (not test_street.any()):
            continue

        mae = np.mean(np.abs(target-prediction))
        # print("mae: ", mae)
        maes.append(mae)

        mse = np.mean((prediction - target) ** 2)
        # print("mse: ", mse)
        mses.append(mse)

        rmse = np.sqrt(mse)
        # print("rmse: ",rmse)
        rmses.append(rmse)

        ssim = 1 - tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(target),
                                  tf.convert_to_tensor(prediction), 1.0))
        ssims.append(ssim)

        sobel_loss = tf.reduce_mean(tf.square(
            sobel(tf.convert_to_tensor(target)) - sobel(tf.convert_to_tensor(prediction))))
        sobels.append(sobel_loss)

    return rmses, maes, mses, ssims, sobels


def predict_shadow(generator, height_path, city, date, zoom, i, j, lat=True, dat=True):
    input_height, input_lat, input_date = load_input_grid(
        height_path, city, date, zoom, i, j)
    input_height, input_lat, input_date = normalize_input(
        input_height, input_lat, input_date)

    input_height = np.array(input_height).reshape(1, 512, 512, 1)
    input_lat = np.array(input_lat).reshape(1, 512, 512, 1)
    input_date = np.array(input_date).reshape(1, 512, 512, 1)

    concat = [input_height]
    if lat:
        concat.append(input_lat)
    if dat:
        concat.append(input_date)

    prediction = generator(concat, training=True)
    prediction = prediction.numpy()[:, 128:-128, 128:-128, :]
    prediction = prediction.reshape(256, 256)

    input_height = input_height[:, 128:-128, 128:-128, :]
    input_height = input_height.reshape(256, 256)
    input_height = (input_height+1)*127.5

    return input_height, prediction


def load_ground_truth(input_height, shadow_path, city, date, zoom, i, j):
    gt_filename = tf.strings.format(
        '{}/{}-{}/{}/{}/{}.png', (shadow_path, city, date, zoom, i, j))
    gt_filename = tf.strings.regex_replace(gt_filename, '\"', "")
    gt = tf.io.read_file(gt_filename)
    gt = tf.io.decode_png(gt)[:, :, 0]
    # remove shadows from buildings in ground truth
    gt = np.where(input_height <= 0, gt, 0)
    # when height is 0, those are pixels that are not buildings and will have shadows.
    # otherwise, they are buildings and will not have shadows. thus should be replaced with 0
    gt = tf.cast(gt, tf.float32)
    gt = (gt / 127.5) - 1.0

    return gt


def test_on_image(generator, height_path, shadow_path, city, date, zoom, i, j, path=None, save=False, lat=True, dat=True):

    def sobel(x): return tf.image.sobel_edges(x)

    def sobel_loss(target, gen_output): return tf.reduce_mean(
        tf.abs(sobel(target) - sobel(gen_output)))

    input_height, prediction = predict_shadow(generator,
                                              height_path, city, date, zoom, i, j, lat=lat, dat=dat)
    gt = load_ground_truth(input_height, shadow_path,
                           city, date, zoom, i, j)

    prediction = np.where(input_height > 0, -1, prediction)

    # prediction = tf.reshape(prediction, (256, 256, 1))
    prediction = tf.cast(prediction, tf.float32)
    prediction = tf.expand_dims(prediction, 0)
    prediction = tf.expand_dims(prediction, -1)

    # gt = tf.reshape(gt, (256, 256, 1))
    gt = tf.cast(gt, tf.float32)
    gt = tf.expand_dims(gt, 0)
    gt = tf.expand_dims(gt, -1)

    plot_comparison(
        input_height, gt[0, :, :, :], prediction[0, :, :, :], path=path, save=save)

    return


def downsample(filters, size, strides=2, apply_batchnorm=True, apply_specnorm=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # result.add(tf.keras.layers.Conv2D(filters, size, strides=2,
    #            padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_specnorm:
        result.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=strides,
                                                                           padding='same', kernel_initializer=initializer, use_bias=False)))

    else:
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides,
                                          padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_batchnorm=True, apply_specnorm=False, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
    #            padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_specnorm:
        result.add(tfa.layers.SpectralNormalization(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                                                    padding='same', kernel_initializer=initializer, use_bias=False)))

    else:
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                   padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def get_generator_arch(g_type, attn=False):

    if (g_type == 'resnet9'):
        down_stack = [
            downsample(64, 4),
            downsample(128, 4, apply_specnorm=attn)
        ]

        up_stack = [
            upsample(64, 4, apply_specnorm=attn),
        ]

    elif (g_type == 'unet'):
        down_stack = [
            # (batch_size, 256, 256, 64)
            downsample(64, 4, apply_batchnorm=False),
            # (batch_size, 128, 128, 128)
            downsample(128, 4, apply_specnorm=attn),
            # (batch_size, 64, 64, 256)
            downsample(256, 4, apply_specnorm=attn),
            # (batch_size, 32, 32, 512)
            downsample(512, 4, apply_specnorm=attn),
            # (batch_size, 16, 16, 512)
            downsample(512, 4, apply_specnorm=attn),
            downsample(512, 4, apply_specnorm=attn),  # (batch_size, 8, 8, 512)
            downsample(512, 4, apply_specnorm=attn),  # (batch_size, 4, 4, 512)
            downsample(512, 4, apply_specnorm=attn),  # (batch_size, 2, 2, 512)
            downsample(512, 4, apply_specnorm=attn),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True, apply_specnorm=attn),
            # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True, apply_specnorm=attn),
            # (batch_size, 8, 8, 1024)
            upsample(512, 4, apply_dropout=True, apply_specnorm=attn),
            # (batch_size, 16, 16, 1024)
            upsample(512, 4, apply_specnorm=attn),
            # (batch_size, 16, 16, 1024)
            upsample(512, 4, apply_specnorm=attn),
            upsample(256, 4, apply_specnorm=attn),  # (batch_size, 32, 32, 512)
            upsample(128, 4, apply_specnorm=attn),  # (batch_size, 64, 64, 256)
            # (batch_size, 128, 128, 128)
            upsample(64, 4, apply_specnorm=attn),
        ]

    elif (g_type == 'paper_model'):
        down_stack = [
            # (batch_size, 128, 128, 64)
            downsample(64, 4, apply_batchnorm=False),
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

    return down_stack, up_stack


def extract_gt_pred(generator, height_path, shadow_path, city, date, zoom, i, j):
    input_height, prediction = predict_shadow(
        generator, height_path, city, date, zoom, i, j)
    gt = load_ground_truth(input_height, shadow_path, city, date, zoom, i, j)

    prediction = prediction * 0.5 + 0.5
    gt = gt * 0.5 + 0.5

    prediction = tf.cast(prediction, tf.float32)
    prediction = tf.expand_dims(prediction, 0)
    prediction = tf.expand_dims(prediction, -1)

    gt = tf.cast(gt, tf.float32)
    gt = tf.expand_dims(gt, 0)
    gt = tf.expand_dims(gt, -1)

    return input_height, gt, prediction


def add_weights(dataset, BATCH_SIZE):
    weights = []
    for i, (input_image, real_image, lat_image, date_image, path) in enumerate(dataset):
        img = input_image[0]
        phi = tf.where(img == -1.0, 0.0, -1.0) + 0.5
        w = skfmm.distance(phi, dx=1)
        w = tf.where(w < 0, 0, w)

        # type cast w to float32
        w = tf.cast(w, tf.float32)

        # normalize w by dividing by 512
        # w = w / 512.0

        # w = tf.expand_dims(w, axis=0)
        weights.append(w)

    weights = tf.data.Dataset.from_tensor_slices(weights)
    weights = weights.batch(BATCH_SIZE)

    dataset = tf.data.Dataset.zip((dataset, weights))

    return dataset
