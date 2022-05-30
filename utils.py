import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from IPython import display
from sklearn.model_selection import train_test_split


def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

def num2deg(xtile, ytile, zoom):
    n = tf.math.pow(2, zoom)
    lon_deg = float(xtile) / float(n) * 360.0 - 180.0
    lat_rad = tf.math.atan(tf.math.sinh(3.14159265359 * (1.0 - 2.0 * float(ytile) / float(n))))
    lat_deg = rad2deg(lat_rad)
    return (lat_deg, lon_deg)

    
def generate_images(model, input_height, input_lat, input_date, target):
    prediction = model([input_height, input_lat, input_date], training=True)    
    plot_comparison(input_height[0], target[0], prediction[0])
    
def plot_result(input_height, prediction):
    plt.figure(figsize=(15, 15))
    
    display_list = [input_height, prediction]
    title = ['Input Image (x10)', 'Predicted Image']

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
    
    plt.show()

def plot_comparison(input_height, target, prediction):
    plt.figure(figsize=(15, 15))
    
    display_list = [input_height, target, prediction]
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

def load_input(height_path, city, zoom, i, j):
    
    # Read and decode an image file to a uint8 tensor
    filename = tf.strings.format('{}/{}/{}/{}/{}.png',(height_path,city,zoom,i,j))
    filename = tf.strings.regex_replace(filename,'\"', "")
    input_image = tf.io.read_file(filename)
    input_image = tf.io.decode_png(input_image)[:,:,0]
    
    input_image = tf.reshape(input_image, (256, 256, 1))
    input_image = tf.cast(input_image, tf.float32)
    
    return input_image
    

def load(height_path, shadow_path, city, date, zoom, i, j):
    
    # Read and decode an image file to a uint8 tensor
    filename = tf.strings.format('{}/{}/{}/{}/{}.png',(height_path,city,zoom,i,j))
    filename = tf.strings.regex_replace(filename,'\"', "")
    input_image = tf.io.read_file(filename)
    input_image = tf.io.decode_png(input_image)[:,:,0]
    
    filename = tf.strings.format('{}/{}-{}/{}/{}/{}.png',(shadow_path,city,date,zoom,i,j))
    filename = tf.strings.regex_replace(filename,'\"', "")
    real_image = tf.io.read_file(filename)
    real_image = tf.io.decode_png(real_image)[:,:,0]
    real_image = tf.experimental.numpy.where(input_image<=0, real_image, 0)    
    
    input_image = tf.reshape(input_image, (256, 256, 1))
    real_image = tf.reshape(real_image, (256, 256, 1))

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    return input_image, real_image


def load_input_grid(height_path, city, date, zoom, i, j):
    all_input = tf.zeros((256*3,256*3,1))
    
    for x in range(-1,2):
        for y in range(-1,2):
            filepath = '%s/%s/%d/%d/%d.png'%(height_path,city,zoom,i+y,j+x)
            if os.path.isfile(filepath):
                iinput = load_input(height_path, city, zoom, i+y, j+x)
                indices = [(xx,yy) for xx in range(256+256*x,256+256*(x+1)) for yy in range(256+256*y,256+256*(y+1))]
                indices = np.array(indices).reshape(256,256,-1)
                all_input = tf.tensor_scatter_nd_update(all_input, indices, iinput)
    
    (latitude, longitude) = num2deg(i, j, zoom)
            
    all_input = all_input[128:-128,128:-128]
    all_lat = tf.ones((512,512), dtype=tf.float32)
    all_lat = tf.math.scalar_mul(float(latitude), all_lat)
    all_lat = tf.reshape(all_lat, (512, 512, 1))
    
    if date == 'winter':
        value = 0
    elif date == 'spring' or date == 'fall':
        value = 1
    else:
        value = 2
        
    all_date = tf.ones((512,512), dtype=tf.float32)
    all_date = tf.math.scalar_mul(float(value), all_date)
    all_date = tf.reshape(all_date, (512, 512, 1))
    
    return all_input, all_lat, all_date

def load_grid(height_path, shadow_path, city, date, zoom, i, j, neighbors):
    all_input = tf.zeros((256*3,256*3,1))
    all_real = tf.zeros((256*3,256*3,1))
    
    count = 0
    for x in range(-1,2):
        for y in range(-1,2):
#             if check_image(height_path, city, date, zoom, i+y, j+x):
                # tensorflow does not support slicing
#                 all_input[256+256*x:256+256*(x+1),256+256*y:256+256*(y+1)], \
#                 all_real[256+256*x:256+256*(x+1),256+256*y:256+256*(y+1)] \
#                 = load(height_path, shadow_path, city, date, zoom, i+y, j+x)
            if neighbors[count] == 'True':
                iinput, real = load(height_path, shadow_path, city, date, zoom, i+y, j+x)
                indices = [(xx,yy) for xx in range(256+256*x,256+256*(x+1)) for yy in range(256+256*y,256+256*(y+1))]
                indices = np.array(indices).reshape(256,256,-1)
                all_input = tf.tensor_scatter_nd_update(all_input, indices, iinput)
                all_real = tf.tensor_scatter_nd_update(all_real, indices, real)
            count+=1
            
            
    (latitude, longitude) = num2deg(i, j, zoom)
            
    all_input = all_input[128:-128,128:-128]
    all_real = all_real[128:-128,128:-128]
    all_lat = tf.ones((512,512), dtype=tf.float32)
    all_lat = tf.math.scalar_mul(float(latitude), all_lat)
    all_lat = tf.reshape(all_lat, (512, 512, 1))
    
    # solve tensorflow's unhashable problem
    if date == 'winter':
        value = 0
    elif date == 'spring' or date == 'fall':
        value = 1
    else:
        value = 2
        
    all_date = tf.ones((512,512), dtype=tf.float32)
    all_date = tf.math.scalar_mul(float(value), all_date)
    all_date = tf.reshape(all_date, (512, 512, 1))
            
    return all_input, all_real, all_lat, all_date


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image, size):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, size, size, 1])

    return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize_input(input_image, lat_image, date_image):
    input_image = (input_image / 127.5) - 1
    lat_image = ((lat_image + 90) / 90.0) - 1
    date_image = date_image - 1

    return input_image, lat_image, date_image

def normalize(input_image, real_image, lat_image, date_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    lat_image = ((lat_image + 90) / 90.0) - 1
    date_image = date_image - 1

    return input_image, real_image, lat_image, date_image

@tf.function()
def random_jitter(input_image, real_image, lat_image, date_image):
    original_size = input_image.shape[0]
    new_size = int(original_size*1.10)
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, new_size, new_size)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image, original_size)

#     if tf.random.uniform(()) > 0.5:
#         Random mirroring
#         input_image = tf.image.flip_left_right(input_image)
#         real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image, lat_image, date_image



def load_image(path):
    height_path, shadow_path, city, date, filename, neighbors = path[0], path[1], path[2], path[3], path[4], path[5:]

    filename = tf.strings.regex_replace(filename,'\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]), int(tf.strings.split(tks[-1],'.')[0])
    
    input_image, real_image, lat_image, date_image = load_grid(height_path, shadow_path, city, date, zoom, i, j, neighbors)
    input_image, real_image, lat_image, date_image = normalize(input_image, real_image, lat_image, date_image)
    
    return input_image, real_image, lat_image, date_image


    
def get_tiles(height_path, shadow_path, cities, dates, zoom, tiles_per_city):
    def get_path(row, city, date):
        values = [height_path, shadow_path, city, date, '%d/%d/%d.png'%(row['zoom'],row['i'],row['j'])]
        for i in range(0,9):
            values.append(str(row[str(i)]))
        return values
    
    all_dataset = []
    for city in cities:
        for date in dates:
            df = pd.read_csv('data/evaluation/%s-%s-%d.csv'%(city,date,zoom))
            df = df.sample(frac=1, random_state=42).reset_index() # shuffle
            df = df.sample(n=min(tiles_per_city,len(df)), random_state=42)
            ds = df.apply(get_path, args=(city, date), axis=1).tolist()
            all_dataset.extend(ds)
            
    return all_dataset

def train_to_tensor(train_dataset, batch_size):
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset

def test_to_tensor(test_dataset, batch_size):
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    test_dataset = test_dataset.map(load_image)
    test_dataset = test_dataset.batch(batch_size)
    
    return test_dataset

    
def get_train_test(height_path, shadow_path, cities, dates, zoom, tiles_per_city, batch_size = 2, train_size = 0.6):
  
    all_dataset = get_tiles(height_path, shadow_path, cities, dates, zoom, tiles_per_city)
    
    train_dataset, test_dataset = train_test_split(all_dataset, train_size=train_size, random_state=42)
    
    train_dataset = train_to_tensor(train_dataset, batch_size)
    test_dataset = test_to_tensor(test_dataset, batch_size)

    return train_dataset, test_dataset


def get_metrics(test_dataset, generator):
    rmses = []
    maes = []
    mses = []
    for test_input, test_target, test_latitude, test_date in test_dataset:
        prediction = generator([test_input, test_latitude, test_date], training=True)
        target = test_target.numpy()[:,128:-128,128:-128,:]
        prediction = prediction.numpy()[:,128:-128,128:-128,:]
        target = target * 0.5 + 0.5
        prediction = prediction * 0.5 + 0.5
        
        mae = np.mean(np.abs(target-prediction))
        maes.append(mae)
        
        mse = np.mean((prediction - target) ** 2)
        mses.append(mse)
        
        rmse = np.sqrt(mse)
        rmses.append(rmse)
     
        
    return rmses, maes, mses
