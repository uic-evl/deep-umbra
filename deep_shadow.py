import time
import skfmm
import os
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

from utils import upsample, downsample, generate_images

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # bce loss


def gan_loss(disc_generated_output):
    return loss_object(tf.ones_like(disc_generated_output), disc_generated_output)


def l1_loss(target, gen_output):
    return tf.reduce_mean(tf.abs(target - gen_output))


def l2_loss(target, gen_output):
    return tf.reduce_mean(tf.square(target - gen_output))


def street_l2_loss(target, gen_output, street_img):
    return tf.reduce_mean(tf.where(street_img > - 1, tf.square(target - gen_output), 0))


huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.AUTO)
def l1_smooth_loss(target, gen_output): return huber(target, gen_output)


def ssim_multiscale_loss(target, gen_output):
    # *** generates nan values ***
    ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(
        target, gen_output, max_val=1.))
    return 1 - ms_ssim


def ssim_loss(target, gen_output):
    ssim = tf.reduce_mean(tf.image.ssim(target, gen_output, 1.))
    return 1 - ssim


def sobel(img): return tf.image.sobel_edges(img)


def sobel_loss(target, gen_output):
    return tf.reduce_mean(
        tf.square(sobel(target) - sobel(gen_output)))


def berhu_loss(target, gen_output):
    c = 1/5 * tf.reduce_max(tf.abs(gen_output - target))
    abs_diff = tf.abs(gen_output - target)
    berhu_loss = tf.reduce_mean(
        tf.where(abs_diff <= c, abs_diff, (abs_diff**2 + c**2)/(2*c)))

    return berhu_loss


def psnr_loss(target, gen_output):
    max_pixel = 1.0
    psnr_val = tf.reduce_mean((10.0 * tf.math.log((max_pixel ** 2) / (
        tf.reduce_mean(tf.square(target - gen_output + 1e-8), axis=-1))))) // 2.303
    # normalize between 0 and 1. max val = 159, min val = 0
    normalized_psnr = psnr_val / 159.0
    return 1 - normalized_psnr


# Using Lambda
def generator_loss(disc_generated_output, gen_output, target, loss_funcs):

    _gan_loss = gan_loss(disc_generated_output)

    _loss = 0
    for loss_func in loss_funcs:
        _loss += loss_func(target, gen_output)

    total_gen_loss = _gan_loss + LAMBDA*_loss

    return total_gen_loss, _gan_loss, _loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# Check if specnorm, instancenorm, batchnorm required in the different models we will try!!
def resblock(filters, size, x, apply_specnorm=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    if (apply_specnorm):
        fx = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters, size, padding='same', kernel_initializer=initializer, use_bias=False))(x)
    else:
        fx = tf.keras.layers.Conv2D(
            filters, size, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    fx = tf.keras.layers.BatchNormalization()(fx)

    if (apply_specnorm):
        fx = tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters, size, padding='same', kernel_initializer=initializer, use_bias=False))(x)
    else:
        fx = tf.keras.layers.Conv2D(
            filters, size, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    fx = tf.keras.layers.BatchNormalization()(fx)

    out = tf.keras.layers.Add()([x, fx])
    out = tf.keras.layers.ReLU()(out)

    return out


class Self_Attention(tf.keras.layers.Layer):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        # Construct the conv layers
        self.query_conv = tf.keras.layers.Conv2D(
            filters=in_dim // 2, kernel_size=1)
        self.key_conv = tf.keras.layers.Conv2D(
            filters=in_dim // 2, kernel_size=1)
        self.value_conv = tf.keras.layers.Conv2D(filters=in_dim, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = tf.Variable(tf.zeros(shape=(1,)), trainable=True)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    @tf.function
    def call(self, x, batch_size=1):
        """
        inputs:
            x: input feature maps (B * C * W * H)
        returns:
            out: self-attention value + input feature
            attention: B * N * N (N is Width*Height)
        """
        m_batchsize, width, height, C = batch_size, x.shape[1], x.shape[2], x.shape[3]

        proj_query = tf.reshape(self.query_conv(
            x), (m_batchsize, -1, width * height))
        proj_query = tf.transpose(proj_query, perm=[0, 2, 1])

        proj_key = tf.reshape(self.key_conv(
            x), (m_batchsize, -1, width * height))  # B * C * N

        energy = tf.matmul(proj_query, proj_key)  # batch matrix-matrix product
        attention = self.softmax(energy)  # B * N * N
        proj_value = tf.reshape(self.value_conv(
            x), (m_batchsize, -1, width * height))  # B * C * N
        # batch matrix-matrix product
        out = tf.matmul(proj_value, tf.transpose(attention, perm=[0, 2, 1]))
        out = tf.reshape(out, (m_batchsize, width, height, C))  # B * C * W * H

        out = self.gamma * out + x
        return out, attention


def Generator(width, height, down_stack, up_stack, latitude=False, date=False, type='unet', attention=False):

    def unet(x):
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        i = 0
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            if (i == 6 and attention):
                self_attn = Self_Attention(in_dim=x.shape[3])
                x, _ = self_attn(x)
            i += 1

        return x

    def resnet9(x):
        # 2 downsampling blocks
        for down in down_stack:
            x = down(x)

        if attention:
            self_attn = Self_Attention(in_dim=x.shape[3])
            x, _ = self_attn(x)

        # 9 residual blocks
        for i in range(9):
            x = resblock(128, 4, x, apply_specnorm=attention)

        # 2 upsampling blocks
        for up in up_stack:
            x = up(x)

        return x

    inputs = tf.keras.layers.Input(shape=[width, height, 1])
    if latitude:
        lat = tf.keras.layers.Input(shape=[width, height, 1])
    if date:
        dat = tf.keras.layers.Input(shape=[width, height, 1])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 512, 512, 1)

    if latitude and date:
        x = tf.keras.layers.concatenate([inputs, lat, dat])
    elif latitude:
        x = tf.keras.layers.concatenate([inputs, lat])
    elif date:
        x = tf.keras.layers.concatenate([inputs, dat])
    else:
        x = inputs

    if (type == 'unet'):
        x = unet(x)
    elif (type == 'resnet9'):
        x = resnet9(x)

    x = last(x)

    ip = [inputs]
    if latitude:
        ip.append(lat)
    if date:
        ip.append(dat)

    return tf.keras.Model(inputs=ip, outputs=x)


def Discriminator(width, height, latitude=False, date=False, type='unet', attention=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[width, height, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[width, height, 1], name='target_image')
    if latitude:
        lat = tf.keras.layers.Input(shape=[width, height, 1], name='latitude')
    if date:
        dat = tf.keras.layers.Input(shape=[width, height, 1], name='date')

    if (latitude and date):
        x = tf.keras.layers.concatenate([inp, tar, lat, dat])
    elif latitude:
        x = tf.keras.layers.concatenate([inp, tar, lat])
    elif date:
        x = tf.keras.layers.concatenate([inp, tar, dat])
    else:
        x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False,
                       apply_specnorm=attention)(x)
    down2 = downsample(128, 4, apply_batchnorm=True, apply_specnorm=attention)(
        down1)
    # add attention
    if (attention):
        self_attn = Self_Attention(in_dim=down2.shape[3])
        down2, _ = self_attn(down2)
    down3 = downsample(256, 4, apply_batchnorm=True, apply_specnorm=attention)(
        down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    down4 = downsample(512, 4, strides=1, apply_batchnorm=True, apply_specnorm=attention)(
        zero_pad1)  # (batch_size, 31, 31, 512)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        down4)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2)

    ip = [inp, tar]
    if latitude:
        ip.append(lat)
    if date:
        ip.append(dat)

    return tf.keras.Model(inputs=ip, outputs=last)


class DeepShadow():

    def __init__(self, width, height, down_stack, up_stack, latitude=True, date=True, loss_funcs=[l1_loss], type='unet', attention=False, model_name='deepshadow'):
        self.lat = latitude
        self.dat = date
        self.loss_funcs = loss_funcs
        self.attention = attention
        self.type = type
        self.model_name = model_name
        self.generator = Generator(
            width, height, down_stack, up_stack, latitude=self.lat, date=self.dat, type=self.type, attention=self.attention)
        self.discriminator = Discriminator(
            width, height, latitude=self.lat, date=self.dat, attention=self.attention)

    def compute_loss(self, test_ds):
        rmse = 0
        for test_input, test_target, _, test_latitude, test_date, _ in test_ds:

            ip = [test_input]
            if self.lat:
                ip.append(test_latitude)
            if self.dat:
                ip.append(test_date)

            prediction = self.generator(ip, training=True)
            prediction = prediction * 0.5 + 0.5
            target = test_target * 0.5 + 0.5

            rmse += tf.sqrt(tf.reduce_mean(tf.square(prediction - target)))

        return rmse / len(test_ds)

    @tf.function
    def train_step(self, input_image, target, street, input_latitude, input_date, summary_writer, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            ip = [input_image]
            if self.lat:
                ip.append(input_latitude)
            if self.dat:
                ip.append(input_date)

            gen_output = self.generator(ip, training=True)

            real = ip[:]
            gen = ip[:]
            real.insert(1, target)
            gen.insert(1, gen_output)

            disc_real_output = self.discriminator(real, training=True)
            disc_generated_output = self.discriminator(gen, training=True)

            gen_total_loss, gen_gan_loss, gen_loss_func = generator_loss(
                disc_generated_output, gen_output, target, self.loss_funcs)

            disc_loss = discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss',
                              gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

        if (step % 1000 == 0):
            tf.print("custom_loss_func: ", gen_loss_func)
            tf.print("gan_loss: ", gen_gan_loss)
            tf.print("disc_loss: ", disc_loss)

    def fit(self, checkpoint_path, train_ds, test_ds, steps, min_delta=0.0001, patience=200):

        g_learning = 1e-4 if self.attention else 2e-4
        d_learning = 4e-4 if self.attention else 2e-4

        self.generator_optimizer = tf.keras.optimizers.Adam(
            g_learning, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            d_learning, beta_1=0.5)

        # logs fit with model name
        summary_writer = tf.summary.create_file_writer(
            "logs/fit_new/" + self.model_name)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_path, max_to_keep=1)

        if test_ds != None:
            example_input, example_target, _, example_lat, example_date, _ = next(
                iter(test_ds.take(1)))

        start = time.time()
        best_loss = np.inf

        for step, (input_image, target, street, latitude, date, _) in train_ds.repeat().take(steps).enumerate():

            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(
                        f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                if test_ds != None:
                    generate_images(self.generator, example_input, example_lat, example_date,
                                    example_target, None, latitude=self.lat, date=self.dat, save=False)

                print(f"Step: {step//1000}k")

            self.train_step(input_image, target, street, latitude,
                            date, summary_writer, step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Early stopping check
            if test_ds is not None and (step + 1) % patience == 0:
                loss = self.compute_loss(test_ds)
                if loss < best_loss:
                    best_loss = loss
                    manager.save()

        loss = self.compute_loss(test_ds)
        if loss < best_loss:
            best_loss = loss
            manager.save()

        # manager.save()

    def restore(self, checkpoint_path):
        g_learning = 1e-4 if self.attention else 2e-4
        d_learning = 4e-4 if self.attention else 2e-4

        self.generator_optimizer = tf.keras.optimizers.Adam(
            g_learning, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            d_learning, beta_1=0.5)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        checkpoint.restore(tf.train.latest_checkpoint(
            checkpoint_path)).expect_partial()

    # Load a checkpoint and save the model in the SavedModel format
    def ckpt2saved(self, checkpoint_path, save_path):
        self.restore(checkpoint_path)
        self.generator.save(save_path)
