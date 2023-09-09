import time
import os
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython import display

from utils import load_grid, normalize, get_tiles

def load_image_height(path):
    height_path, shadow_path, city, date, filename, neighbors = path[0], path[1], path[2], path[3], path[4], path[5:]

    filename = tf.strings.regex_replace(filename,'\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]), int(tf.strings.split(tks[-1],'.')[0])
    
    input_image, real_image, lat_image, date_image = load_grid(height_path, shadow_path, city, date, zoom, i, j, neighbors)
    input_image, real_image, lat_image, date_image = normalize(input_image, real_image, lat_image, date_image)
    
    return input_image, real_image, lat_image, date_image

def load_image_shadow(path):
    height_path, shadow_path, city, date, filename, neighbors = path[0], path[1], path[2], path[3], path[4], path[5:]

    filename = tf.strings.regex_replace(filename,'\\\\', "/")
    tks = tf.strings.split(filename, '/')
    zoom, i, j = int(tks[-3]), int(tks[-2]), int(tf.strings.split(tks[-1],'.')[0])
    
    input_image, real_image, lat_image, date_image = load_grid(height_path, shadow_path, city, date, zoom, i, j, neighbors)
    input_image, real_image, lat_image, date_image = normalize(input_image, real_image, lat_image, date_image)
    
    return real_image, input_image, lat_image, date_image

def train_to_tensor(train_dataset, batch_size, height=True):
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)

    if(height):
        train_dataset = train_dataset.map(load_image_height, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.map(load_image_shadow, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset

def test_to_tensor(test_dataset, batch_size, height=True):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

    if(height):
        test_dataset = test_dataset.map(load_image_height)
    else:
        test_dataset = test_dataset.map(load_image_shadow)

    test_dataset = test_dataset.batch(batch_size)
    
    return test_dataset

def get_train_test(height_path, shadow_path, cities, dates, zoom, tiles_per_city, batch_size = 2, train_size = 0.6):

    all_dataset = get_tiles(height_path, shadow_path, cities, dates, zoom, tiles_per_city)

    height_dataset = shadow_dataset = all_dataset

    train_height, test_height = train_test_split(height_dataset, train_size=train_size, random_state=42)
    train_shadow, test_shadow = train_test_split(shadow_dataset, train_size=train_size, random_state=42)
    
    train_height = train_to_tensor(train_height, batch_size, height = True)
    test_height = test_to_tensor(test_height, batch_size, height = True)

    train_shadow = train_to_tensor(train_shadow, batch_size, height = False)
    test_shadow = test_to_tensor(test_shadow, batch_size, height = False)

    return train_height, test_height, train_shadow, test_shadow


# from utils import generate_images

LAMBDA = 10
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output[0]), disc_real_output[0])

    generated_loss = loss_object(tf.zeros_like(disc_generated_output[0]), disc_generated_output[0])

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_generated_output):
    gan_loss = loss_object(tf.ones_like(disc_generated_output[0]), disc_generated_output[0])

    return gan_loss

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image[0] - cycled_image[0]))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image[0] - same_image[0]))
  return LAMBDA * 0.5 * loss

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    
    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator(width, height, norm_type='batchnorm'):
    inputs = tf.keras.layers.Input(shape=[width, height, 1])
    lat = tf.keras.layers.Input(shape=[width,height,1])
    dat = tf.keras.layers.Input(shape=[width,height,1])

    down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
      downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
      downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
      downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
    ]

    up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = tf.keras.layers.concatenate([inputs, lat, dat])
    # x= inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs, lat, dat], outputs=[x, lat, dat])
    # return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator(width, height, norm_type='batchnorm'):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[width, height, 1], name='input_image')
    lat = tf.keras.layers.Input(shape=[width, height, 1], name='latitude')
    dat = tf.keras.layers.Input(shape=[width, height, 1], name='date')

    x = tf.keras.layers.concatenate([inp, lat, dat])  # (batch_size, 256, 256, channels*2)
    # x = inp

    down1 = downsample(64, 4, norm_type, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, lat, dat], outputs=[last, lat, dat])
    # return tf.keras.Model(inputs=inp, outputs=last)
    

def generate_images(model, test_input):
    prediction = model([test_input[0], test_input[2], test_input[3]])
    # prediction = model(test_input[0])
        
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0][0], prediction[0][0], test_input[1][0]]
    title = ['Input Image', 'Predicted Image', 'Ground Truth']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

class CycleGAN():
    def __init__(self, width, height):
        self.generator_g = Generator(width, height, norm_type='instancenorm')
        self.generator_f = Generator(width, height, norm_type='instancenorm')

        self.discriminator_x = Discriminator(width, height, norm_type='instancenorm')
        self.discriminator_y = Discriminator(width, height, norm_type='instancenorm')

    @tf.function
    def train_step(self, real_x, real_y, summary_writer, step):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)
            
            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
        
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                                self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                                self.generator_f.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                    self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                    self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                    self.generator_f.trainable_variables))
        
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                        self.discriminator_x.trainable_variables))
        
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                        self.discriminator_y.trainable_variables))
        

        with summary_writer.as_default():
            tf.summary.scalar('total_gen_f_loss', total_gen_f_loss, step=step//5)
            tf.summary.scalar('total_gen_g_loss', total_gen_g_loss, step=step//5)
            tf.summary.scalar('disc_x_loss', disc_x_loss, step=step//5)
            tf.summary.scalar('disc_y_loss', disc_y_loss, step=step//5)
            
    def fit(self, checkpoint_path, train_height, train_shadow, steps):

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        summary_writer = tf.summary.create_file_writer("logs/fit/cyclegan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        checkpoint = tf.train.Checkpoint(generator_g = self.generator_g,
                           generator_f = self.generator_f,
                           discriminator_x = self.discriminator_x,
                           discriminator_y = self.discriminator_y,
                           generator_g_optimizer = self.generator_g_optimizer,
                           generator_f_optimizer = self.generator_f_optimizer,
                           discriminator_x_optimizer = self.discriminator_x_optimizer,
                           discriminator_y_optimizer = self.discriminator_y_optimizer)
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=1)

        if train_height != None:
            sample_height = next(iter(train_height))

        for epoch in range(steps):
            start = time.time()

            n = 0
            for image_x, image_y in tf.data.Dataset.zip((train_height, train_shadow)):
                self.train_step([image_x[0], image_x[2], image_x[3]], [image_y[0], image_y[2], image_y[3]], summary_writer, epoch)
                if n % 10 == 0:
                    print('.', end='', flush=True)
                n += 1

            display.clear_output(wait=True)

            # Using a consistent image (sample_height) so that the progress of the model
            # is clearly visible.
            generate_images(self.generator_g, sample_height)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))
            
        ckpt_save_path = manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        
    def restore(self, checkpoint_path):

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        checkpoint = tf.train.Checkpoint(generator_g = self.generator_g,
                           generator_f = self.generator_f,
                           discriminator_x = self.discriminator_x,
                           discriminator_y = self.discriminator_y,
                           generator_g_optimizer = self.generator_g_optimizer,
                           generator_f_optimizer = self.generator_f_optimizer,
                           discriminator_x_optimizer = self.discriminator_x_optimizer,
                           discriminator_y_optimizer = self.discriminator_y_optimizer)
        
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    # Load a checkpoint and save the model in the SavedModel format
    def ckpt2saved(self, checkpoint_path, save_path):
        self.restore(checkpoint_path)
        self.generator.save(save_path)


def get_metrics(test_dataset, generator):
    
    rmses = []
    maes = []
    mses = []
    for test_input in test_dataset:
        prediction = generator([test_input[0], test_input[2], test_input[3]], training=False)
        
        
        # target = test_input[1][0].numpy()[:,128:-128,128:-128,:]
        target = test_input[1].numpy()[:,128:-128,128:-128,:]

        # prediction = prediction[0][0].numpy()[:,128:-128,128:-128,:]
        prediction = prediction[0].numpy()[:,128:-128,128:-128,:]

        target = target * 0.5 + 0.5
        prediction = prediction * 0.5 + 0.5
        
        mae = np.mean(np.abs(target-prediction))
        maes.append(mae)
        
        mse = np.mean((prediction - target) ** 2)
        mses.append(mse)
        
        rmse = np.sqrt(mse)
        rmses.append(rmse)
     
        
    return rmses, maes, mses
