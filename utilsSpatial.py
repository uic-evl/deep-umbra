import os
import re
import pathlib
import time
import itertools
import glob
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from IPython import display
from sklearn.model_selection import train_test_split

from generatorSpatial import *

def generate_images(model, test_input, latitude_matrix, tar):
    #prediction = model(test_input, training=True)
    prediction = model([test_input, latitude_matrix], training=True)
    plt.figure(figsize=(15, 15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image (x10)', 'Ground Truth', 'Predicted Image']

    plt.subplot(1, 3, 1)
    plt.title(title[0])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow((display_list[0] * 0.5 + 0.5) * 10)
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
    
    plt.show()

#TODO the load function must also load the latitude matrix attached to the image
def loadSpatial(height_path, shadow_path, zoom, i, j):
    
    # Read and decode an image file to a uint8 tensor
    filename = tf.strings.format('{}/{}/{}/{}.png',(height_path,zoom,i,j))
    filename = tf.strings.regex_replace(filename,'\"', "")
    input_image = tf.io.read_file(filename)
    input_image = tf.io.decode_png(input_image)[:,:,0]
#     input_image = input_image.numpy()
#     input_image = tf.reshape(input_image, (*input_image.shape, 1))
#     input_image = tf.image.grayscale_to_rgb(input_image)
    
    filename = tf.strings.format('{}/{}/{}/{}.png',(shadow_path,zoom,i,j))
    filename = tf.strings.regex_replace(filename,'\"', "")
    real_image = tf.io.read_file(filename)
    real_image = tf.io.decode_png(real_image)[:,:,0]
#     real_image = real_image.numpy()
#     real_image[input_image>0] = 0
    real_image = tf.experimental.numpy.where(input_image<=0, real_image, 0)    
    
#     input_image = tf.convert_to_tensor(input_image)
#     input_image = tf.reshape(input_image, (*input_image.shape, 1))
    input_image = tf.reshape(input_image, (256, 256, 1))
    input_image = tf.image.grayscale_to_rgb(input_image)
    
#     real_image = tf.convert_to_tensor(real_image)
#     real_image = tf.reshape(real_image, (*real_image.shape, 1))
    real_image = tf.reshape(real_image, (256, 256, 1))
    real_image = tf.image.grayscale_to_rgb(real_image)

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    #if tf.random.uniform(()) > 0.5:
    #    # Random mirroring
    #    input_image = tf.image.flip_left_right(input_image)
    #    real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

#TODO the load function must also load the latitude matrix attached to the image
def load_image_train_spatial(paths):
    height_path, shadow_path, filename = paths[0], paths[1], paths[2]
    
    filename = tf.strings.regex_replace(filename,'\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]), int(tf.strings.split(tks[-1],'.')[0])
    
    input_image, real_image = loadSpatial(height_path, shadow_path, zoom, i, j)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    
#     input_image = tf.reshape(input_image, (1, *input_image.shape))
#     real_image = tf.reshape(real_image, (1, *real_image.shape))
    
    return input_image, real_image

#TODO the load function must also load the latitude matrix attached to the image
def load_image_test_spatial(paths):
    height_path, shadow_path, filename = paths[0], paths[1], paths[2]

    filename = tf.strings.regex_replace(filename,'\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]), int(tf.strings.split(tks[-1],'.')[0])
    
    input_image, real_image = loadSpatial(height_path, shadow_path, zoom, i, j)
    input_image, real_image = resize(input_image, real_image,IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    
#     input_image = tf.reshape(input_image, (1, *input_image.shape))
#     real_image = tf.reshape(real_image, (1, *real_image.shape))

    return input_image, real_image

def get_train_test_spatial(height_path, shadow_path, city, date, zoom):

    def get_path(row):
        return (height_path, shadow_path, '%s/%d/%d/%d.png'%(shadow_path,row['zoom'],row['i'],row['j']))

    df = pd.read_csv('data/%s-%s-%d.csv'%(city,date,zoom))
    all_dataset = df.apply(get_path, axis=1).tolist()

    train_dataset, test_dataset = train_test_split(all_dataset, train_size=0.6, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(load_image_train_spatial,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    test_dataset = test_dataset.map(load_image_test_spatial)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset