import time
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display

from generator import *
from discriminator import *

from utils import *

@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, input_image, target, input_latitude, input_date, summary_writer, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_image, input_latitude, input_date], training=True)

        disc_real_output = discriminator([input_image, target, input_latitude, input_date], training=True)
        disc_generated_output = discriminator([input_image, gen_output, input_latitude, input_date], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        

    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(checkpoint_path, generator, discriminator, train_ds, test_ds, steps):

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    example_input, example_target, example_date, example_lat = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target, latitude, date) in train_ds.repeat().take(steps).enumerate(): 
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_lat, example_date, example_target)
            print(f"Step: {step//1000}k")

        train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, input_image, target, latitude, date, summary_writer, step)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

def restore(checkpoint_path, generator, discriminator):
    
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    return checkpoint
