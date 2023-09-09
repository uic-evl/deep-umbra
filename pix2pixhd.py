
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from IPython import display
import datetime
import time
from utils import *

from deep_shadow import *

LAMBDA = 10

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # bce loss


def generator_loss_pix2pix(disc_generated_output, disc_generated_output2, disc_generated_output4, gen_output, target, loss_funcs):

    _gan_loss = gan_loss(disc_generated_output)
    _gan_loss += gan_loss(disc_generated_output2)
    _gan_loss += gan_loss(disc_generated_output4)

    _loss = 0
    for loss_func in loss_funcs:
        _loss += loss_func(target, gen_output)

    total_gen_loss = _gan_loss + LAMBDA*_loss

    return total_gen_loss, _gan_loss, _loss


class pix2pixHD():

    def __init__(self, width, height, down_stack, up_stack, latitude=False, date=False, loss_funcs=[l1_loss], type='pix2pixHD', attention=False, model_name='pix2pixHD'):
        self.lat = latitude
        self.dat = date
        self.loss_funcs = loss_funcs
        self.type = type
        self.attention = attention
        self.model_name = model_name
        self.generator = Generator(
            width, height, down_stack, up_stack, self.lat, self.dat, type=self.type, attention=self.attention)
        self.discriminator = Discriminator(
            width, height, self.lat, self.dat, type=type, attention=self.attention)
        self.discriminator2 = Discriminator(
            int(width/2), int(height/2), self.lat, self.dat,  type=type, attention=self.attention)
        self.discriminator4 = Discriminator(
            int(width/4), int(height/4), self.lat, self.dat,  type=type, attention=self.attention)

    def compute_loss(self, test_ds):
        # ssim_ls = 0
        rmse = 0
        for test_input, test_target, test_latitude, test_date, _ in test_ds:
            ip = [test_input]
            if self.lat:
                ip.append(test_latitude)
            if self.dat:
                ip.append(test_date)

            prediction = self.generator(ip, training=True)
            prediction = prediction * 0.5 + 0.5
            target = test_target * 0.5 + 0.5

            # ssim_ls += (1 - tf.reduce_mean(tf.image.ssim(prediction, target, 1.0)))

            # rmse for two images
            rmse += tf.sqrt(tf.reduce_mean(tf.square(prediction - target)))

        # print('Avg RMSE on test set: ', (rmse / len(test_ds)).numpy())

        return rmse / len(test_ds)

    @tf.function
    def train_step(self, input_image, target, input_latitude, input_date, summary_writer, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as disc_tape2, tf.GradientTape() as disc_tape4:

            # reshape input image by scaling down 2x
            input_image2 = tf.image.resize(
                input_image, [int(input_image.shape[1]/2), int(input_image.shape[2]/2)])
            input_image4 = tf.image.resize(
                input_image, [int(input_image.shape[1]/4), int(input_image.shape[2]/4)])

            target2 = tf.image.resize(
                target, [int(target.shape[1]/2), int(target.shape[2]/2)])

            target4 = tf.image.resize(
                target, [int(target.shape[1]/4), int(target.shape[2]/4)])

            ip = [input_image]
            ip2 = [input_image2]
            ip4 = [input_image4]

            if (self.lat):
                input_latitude2 = tf.image.resize(
                    input_latitude, [int(input_latitude.shape[1]/2), int(input_latitude.shape[2]/2)])
                input_latitude4 = tf.image.resize(
                    input_latitude, [int(input_latitude.shape[1]/4), int(input_latitude.shape[2]/4)])

                ip.append(input_latitude)
                ip2.append(input_latitude2)
                ip4.append(input_latitude4)

            if (self.dat):
                input_date2 = tf.image.resize(
                    input_date, [int(input_date.shape[1]/2), int(input_date.shape[2]/2)])
                input_date4 = tf.image.resize(
                    input_date, [int(input_date.shape[1]/4), int(input_date.shape[2]/4)])

                ip.append(input_date)
                ip2.append(input_date2)
                ip4.append(input_date4)

            gen_output = self.generator(
                ip, training=True)
            gen_output2 = tf.image.resize(
                gen_output, [int(gen_output.shape[1]/2), int(gen_output.shape[2]/2)])
            gen_output4 = tf.image.resize(
                gen_output, [int(gen_output.shape[1]/4), int(gen_output.shape[2]/4)])

            real = ip[:]
            gen = ip[:]

            real.insert(1, target)
            gen.insert(1, gen_output)

            real2 = ip2[:]
            gen2 = ip2[:]

            real2.insert(1, target2)
            gen2.insert(1, gen_output2)

            real4 = ip4[:]
            gen4 = ip4[:]
            real4.insert(1, target4)
            gen4.insert(1, gen_output4)

            disc_real_output = self.discriminator(real, training=True)
            disc_generated_output = self.discriminator(gen, training=True)

            disc_real_output2 = self.discriminator2(real2, training=True)
            disc_generated_output2 = self.discriminator2(
                gen2, training=True)

            disc_real_output4 = self.discriminator4(real4, training=True)
            disc_generated_output4 = self.discriminator4(
                gen4, training=True)

            # feature matching loss for all 3 discriminators and all 4 feature maps

            fm_loss_d, fm_loss_d2, fm_loss_d4 = 0, 0, 0
            for i in range(4):
                fm_loss_d += tf.reduce_mean(
                    tf.abs(disc_real_output[i] - disc_generated_output[i]))
                fm_loss_d2 += tf.reduce_mean(
                    tf.abs(disc_real_output2[i] - disc_generated_output2[i]))
                fm_loss_d4 += tf.reduce_mean(
                    tf.abs(disc_real_output4[i] - disc_generated_output4[i]))

            fm_loss_d /= 4
            fm_loss_d2 /= 4
            fm_loss_d4 /= 4

            fm_loss = fm_loss_d + fm_loss_d2 + fm_loss_d4

            # [3] is the last feature map
            disc_loss = discriminator_loss(
                disc_real_output[3], disc_generated_output[3])

            disc_loss2 = discriminator_loss(
                disc_real_output2[3], disc_generated_output2[3])

            disc_loss4 = discriminator_loss(
                disc_real_output4[3], disc_generated_output4[3])

            total_loss, gen_gan_loss, gen_loss_func = generator_loss_pix2pix(
                disc_generated_output[3], disc_generated_output2[3], disc_generated_output4[3],  gen_output, target, self.loss_funcs)

            gen_total_loss = total_loss + (10 * fm_loss)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)

        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        discriminator_gradients2 = disc_tape2.gradient(
            disc_loss2, self.discriminator2.trainable_variables)

        discriminator_gradients4 = disc_tape4.gradient(
            disc_loss4, self.discriminator4.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))
        self.discriminator_optimizer2.apply_gradients(
            zip(discriminator_gradients2, self.discriminator2.trainable_variables))
        self.discriminator_optimizer4.apply_gradients(
            zip(discriminator_gradients4, self.discriminator4.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss',
                              gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            # tf.summary.scalar('gen_def_loss', gen_loss_func, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
            tf.summary.scalar('disc_loss2', disc_loss2, step=step//1000)
            tf.summary.scalar('disc_loss4', disc_loss4, step=step//1000)
            tf.summary.scalar('total_disc_loss', disc_loss +
                              disc_loss2 + disc_loss4, step=step//1000)
            tf.summary.scalar('fm_loss', fm_loss, step=step//1000)

        if (step % 1000 == 0):
            tf.print("fm_loss: ", fm_loss)
            tf.print("gan_loss: ", gen_gan_loss)
            tf.print("total_disc_loss: ", disc_loss + disc_loss2 + disc_loss4)

    def fit(self, checkpoint_path, train_ds, test_ds, steps,  min_delta=0.0001, patience=50):

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_optimizer2 = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_optimizer4 = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

        # logs fit with model name
        summary_writer = tf.summary.create_file_writer(
            "logs/fit/" + self.model_name)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         discriminator_optimizer2=self.discriminator_optimizer2,
                                         discriminator_optimizer4=self.discriminator_optimizer4,
                                         generator=self.generator,
                                         discriminator=self.discriminator,
                                         discriminator2=self.discriminator2,
                                         discriminator4=self.discriminator4)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_path, max_to_keep=1)

        if test_ds != None:
            example_input, example_target, example_date, example_lat, _ = next(
                iter(test_ds.take(1)))

        start = time.time()
        best_loss = np.inf
        # steps_without_improvement = 0

        for step, (input_image, target, latitude, date, _) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(
                        f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                if test_ds != None:
                    # print(self.lat, self.dat)
                    generate_images(self.generator, example_input, example_lat, example_date,
                                    example_target, None, latitude=self.lat, date=self.dat, save=False)

                print(f"Step: {step//1000}k")

            self.train_step(input_image, target, latitude,
                            date, summary_writer, step)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 10k steps
            # if (step + 1) % 10000 == 0:
            #     manager.save()

            # Early stopping check
            if test_ds is not None and (step + 1) % patience == 0:
                loss = self.compute_loss(test_ds)
                if loss < best_loss:
                    best_loss = loss
                    # steps_without_improvement = 0
                    manager.save()
                # else:
                #     steps_without_improvement += 1
                #     if steps_without_improvement >= patience:
                #         print(
                #             f"\nEarly stopping at step {step+1} because the loss did not improve.")
                #         break

        loss = self.compute_loss(test_ds)
        if loss < best_loss:
            best_loss = loss
            manager.save()

        # manager.save()

    def restore(self, checkpoint_path):
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_optimizer2 = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_optimizer4 = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         discriminator_optimizer2=self.discriminator_optimizer2,
                                         discriminator_optimizer4=self.discriminator_optimizer4,
                                         generator=self.generator,
                                         discriminator=self.discriminator,
                                         discriminator2=self.discriminator2,
                                         discriminator4=self.discriminator4)

        checkpoint.restore(tf.train.latest_checkpoint(
            checkpoint_path)).expect_partial()

    # Load a checkpoint and save the model in the SavedModel format
    def ckpt2saved(self, checkpoint_path, save_path):
        self.restore(checkpoint_path)
        self.generator.save(save_path)
