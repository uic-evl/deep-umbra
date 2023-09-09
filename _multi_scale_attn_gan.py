import tensorflow as tf
from utils import *
from deep_shadow import downsample, upsample, resblock, discriminator_loss, l1_loss, ssim_loss, sobel_loss, Self_Attention

LAMBDA = 10000
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # bce loss

def gan_loss(disc_generated_output): return loss_object(
    tf.ones_like(disc_generated_output), disc_generated_output)

def generator_loss(disc_generated_output, gen_output, target, loss_funcs, orig_img=False):

    _gan_loss = gan_loss(disc_generated_output)

    _loss = 0

    if orig_img:
        for loss_func in loss_funcs:
            _loss += loss_func(target, gen_output)

    total_gen_loss = _gan_loss + LAMBDA*_loss

    return total_gen_loss, _gan_loss, _loss

def Generator(width, height, latitude=False, date=False, type='unet', attention=False, add_img=False):

    def unet(x, down_stack, up_stack):
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
    
    def resnet9(x, filters, down_stack, up_stack):
        # 2 downsampling blocks
        for down in down_stack:
            x = down(x)

        if attention:
            self_attn = Self_Attention(in_dim=x.shape[3])
            x, _ = self_attn(x)

        # 9 residual blocks
        for i in range(9):
            x = resblock(filters*2, 4, x, apply_specnorm=attention)

        # 2 upsampling blocks
        for up in up_stack:
            x = up(x)

        return x

    def multi_scale_gan(x, type='unet'):
        filters = x.shape[1] // 8
        # iter = 3

        if(type == 'unet'):
            # for i in range(4): 
            down_stack = [downsample(filters, 4, apply_batchnorm=False), downsample(filters*2, 4), downsample(filters*4, 4)] + [downsample (filters * 8, 4) for _ in range(iter)]
            up_stack = [upsample(filters * 8, 4, apply_dropout=True) for _ in range(iter-2)] + [upsample(filters * 8, 4) for _ in range(1)] + [upsample(filters*4, 4), upsample(filters*2, 4), upsample(filters, 4)]

            x = unet(x, down_stack, up_stack)
            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(1, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')
            x = last(x)

            filters = int(filters * 2)

        elif(type == 'resnet9'):
            down_stack = [downsample(filters, 4), downsample(filters*2, 4, apply_specnorm=attention)]
            up_stack = [upsample(filters, 4, apply_specnorm=attention)]

            x = resnet9(x, filters, down_stack, up_stack)
            initializer = tf.random_normal_initializer(0., 0.02)
            last = tf.keras.layers.Conv2DTranspose(1, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')
            x = last(x)

        return x
    
    inputs = tf.keras.layers.Input(shape=[width, height, 1])
    if latitude:
        lat = tf.keras.layers.Input(shape=[width, height, 1])
    if date:
        dat = tf.keras.layers.Input(shape=[width, height, 1])
    if add_img:
        img = tf.keras.layers.Input(shape=[width, height, 1])

    concat_inputs = [inputs]
    if latitude:
        concat_inputs.append(lat)
    if date:
        concat_inputs.append(dat)
    if add_img:
        concat_inputs.append(img)

    x = tf.keras.layers.concatenate(concat_inputs) if len(concat_inputs) > 1 else inputs 

    out = multi_scale_gan(x, type=type)

    ip = [inputs]
    if latitude:
        ip.append(lat)
    if date:
        ip.append(dat)
    if add_img:
        ip.append(img)

    return tf.keras.Model(inputs=ip, outputs=out)

def Discriminator(width, height, latitude=False, date=False, type='unet', attention=False, add_img=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[width, height, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[width, height, 1], name='target_image')
    if latitude:
        lat = tf.keras.layers.Input(shape=[width, height, 1], name='latitude')
    if date:
        dat = tf.keras.layers.Input(shape=[width, height, 1], name='date')
    if add_img:
        img = tf.keras.layers.Input(shape=[width, height, 1], name='add_img')

    concat_inputs = [inp, tar]
    if latitude:
        concat_inputs.append(lat)
    if date:
        concat_inputs.append(dat)
    if add_img:
        concat_inputs.append(img)

    x = tf.keras.layers.concatenate(concat_inputs)

    filters = height // 8

    down1 = downsample(filters, 4, apply_batchnorm=False, apply_specnorm=attention)(x) # 64
    down2 = downsample(filters * 2, 4, apply_batchnorm=True, apply_specnorm=attention)(down1) # 128

    # add attention
    if (attention):
        self_attn = Self_Attention(in_dim=down2.shape[3])
        down2, _ = self_attn(down2)
    
    down3 = downsample(filters * 4, 4, apply_batchnorm=True, apply_specnorm=attention)(down2) # 256

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    down4 = downsample(filters * 8, 4, strides=1, apply_batchnorm=True, apply_specnorm=attention)(zero_pad1) # 512

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        down4)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2)

    ip = [inp, tar]
    if latitude:
        ip.append(lat)
    if date:
        ip.append(dat)
    if add_img:
        ip.append(img)

    return tf.keras.Model(inputs=ip, outputs=last)

class MultiGAN():

    def __init__(self, width, height, latitude=True, date=True, loss_funcs=[l1_loss], type='resnet', attention=False, model_name='multi_scale_cgan'):
        self.width = width
        self.height = height
        self.lat = latitude
        self.dat = date
        self.loss_funcs = loss_funcs
        self.attention = attention
        self.type = type
        self.model_name = model_name        
        self.generators = [Generator(64*i, 64*i, latitude=self.lat, date=self.dat, 
                                    type=self.type, attention=self.attention, add_img = True if i!=1 else False) for i in [1, 2, 4, 8]]
        self.discriminators = [Discriminator(64*i, 64*i, latitude=self.lat, date=self.dat,
                                            type=self.type, attention=self.attention, add_img = True if i!=1 else False) for i in [1, 2, 4, 8]]

    def compute_loss(self, test_ds):
        rmse = 0
        for test_input, test_target, test_latitude, test_date, _ in test_ds:

            scales = [8, 4, 2 ,1]
            
            ip_images = [test_input if i == 1 else tf.image.resize(
                test_input, [self.height // i, self.width // i]) for i in scales] 
            if(self.lat):
                ip_lats = [test_latitude if i == 1 else tf.image.resize(
                    test_latitude, [self.height // i, self.width // i]) for i in scales]
            if(self.dat):
                ip_dates = [test_date if i == 1 else tf.image.resize(
                    test_date, [self.height // i, self.width // i]) for i in scales]
            
            ip = []
            for i in range(4):
                ip.append([ip_images[i]])
                if(self.lat): ip[i].append(ip_lats[i])
                if(self.dat): ip[i].append(ip_dates[i])
            
            predictions = []
            for i in range(4):
                if(i > 0):
                    ip[i].append(tf.image.resize(predictions[i-1], [predictions[i-1].shape[1]*2, predictions[i-1].shape[2]*2]))
                
                predictions.append(self.generators[i](ip[i], training=True))

            prediction = predictions[3] * 0.5 + 0.5
            target = test_target * 0.5 + 0.5

            rmse += tf.sqrt(tf.reduce_mean(tf.square(prediction - target)))

        return rmse / len(test_ds)

    @tf.function
    def train_step(self, input_image, target, input_latitude, input_date, summary_writer, step):
        
        gen_tapes = [tf.GradientTape() for _ in range(4)]
        disc_tapes = [tf.GradientTape() for _ in range(4)]

        with gen_tapes[0], disc_tapes[0], gen_tapes[1], disc_tapes[1], gen_tapes[2], disc_tapes[2], gen_tapes[3], disc_tapes[3]:
            
            scales = [8, 4, 2 ,1]
            
            ip_images = [input_image if i == 1 else tf.image.resize(
                input_image, [self.height // i, self.width // i]) for i in scales] 
            if(self.lat):
                ip_lats = [input_latitude if i == 1 else tf.image.resize(
                    input_latitude, [self.height // i, self.width // i]) for i in scales]
            if(self.dat):
                ip_dates = [input_date if i == 1 else tf.image.resize(
                    input_date, [self.height // i, self.width // i]) for i in scales]
            target_imgs = [target if i == 1 else tf.image.resize(
                target, [self.height // i, self.width // i]) for i in scales]
            
            ip = []
            for i in range(4):
                ip.append([ip_images[i]])
                if(self.lat): ip[i].append(ip_lats[i])
                if(self.dat): ip[i].append(ip_dates[i])

            gen_outputs = []
            for i in range(4):
                if(i > 0):
                    # upscale gen_outputs[i-1] by 2 and then append to ip[i]
                    ip[i].append(tf.image.resize(gen_outputs[i-1], [gen_outputs[i-1].shape[1]*2, gen_outputs[i-1].shape[2]*2]))
                gen_outputs.append(self.generators[i](ip[i], training=True))

            disc_real_outputs , disc_gen_outputs = [], []
            gen_total_losses, gen_gan_losses, gen_custom_losses = [], [], []
            disc_losses = []

            for i in range(4):
                real = ip[i][:]
                gen = ip[i][:]
                
                real.insert(1, target_imgs[i])
                gen.insert(1, gen_outputs[i])

                disc_real_outputs.append(self.discriminators[i](real, training=True))
                disc_gen_outputs.append(self.discriminators[i](gen, training=True))
                
                orig = True if(i==3) else False
                gen_total_loss, gen_gan_loss, gen_loss_func = generator_loss(
                    disc_gen_outputs[i], gen_outputs[i], target_imgs[i], self.loss_funcs, orig_img=orig)
                
                gen_total_losses.append(gen_total_loss)
                gen_gan_losses.append(gen_gan_loss)
                gen_custom_losses.append(gen_loss_func)

                disc_losses.append(discriminator_loss(
                    disc_real_outputs[i], disc_gen_outputs[i]))
            
            generator_gradients = [gen_tapes[i].gradient(
                gen_total_losses[i], self.generators[i].trainable_variables) for i in range(4)]
            discriminator_gradients = [disc_tapes[i].gradient(
                disc_losses[i], self.discriminators[i].trainable_variables) for i in range(4)]

        for i in range(4):
            self.generator_optimizers[i].apply_gradients(
                zip(generator_gradients[i], self.generators[i].trainable_variables))
            self.discriminator_optimizers[i].apply_gradients(
                zip(discriminator_gradients[i], self.discriminators[i].trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_gan_loss', gen_gan_losses[3], step=step//1000)
            tf.summary.scalar('gen_total_loss', gen_total_losses[3], step=step//1000)
            tf.summary.scalar('disc_loss', disc_losses[3], step=step//1000)

        if (step % 1000 == 0):
            tf.print('gan_loss', gen_gan_losses[3])
            tf.print('disc_loss', disc_losses[3])
            tf.print('gen_total_loss', gen_total_losses[3])

    def fit(self, checkpoint_path, train_ds, test_ds, steps, min_delta=0.0001, patience=50):

        g_learning = 1e-4 if self.attention else 2e-4
        d_learning = 4e-4 if self.attention else 2e-4

        self.generator_optimizers = [tf.keras.optimizers.Adam(
            g_learning, beta_1=0.5) for _ in range(4)]
        self.discriminator_optimizers = [tf.keras.optimizers.Adam(
            d_learning, beta_1=0.5) for _ in range(4)] 

        # logs fit with model name
        summary_writer = tf.summary.create_file_writer(
            "logs/fit_new/" + self.model_name)
        
        checkpoint = tf.train.Checkpoint(generator_optimizers=self.generator_optimizers,
                                         discriminator_optimizers=self.discriminator_optimizers,
                                         generators=self.generators,
                                         discriminators=self.discriminators)
        
        manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_path, max_to_keep=1)

        if test_ds != None:
            example_input, example_target, example_date, example_lat, _ = next(
                iter(test_ds.take(1)))

        start = time.time()
        best_loss = np.inf

        for step, (input_image, target, latitude, date, _) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(
                        f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                if test_ds != None:
                    generate_images(self.height, self.width, self.lat, self.dat, example_input, example_lat, example_date, example_target, self.generators)

                print(f"Step: {step//1000}k")

            self.train_step(input_image, target, latitude,
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

    def restore(self, checkpoint_path):
        g_learning = 1e-4 if self.attention else 2e-4
        d_learning = 4e-4 if self.attention else 2e-4

        self.generator_optimizers = [tf.keras.optimizers.Adam(
            g_learning, beta_1=0.5) for _ in range(4)]
        self.discriminator_optimizers = [tf.keras.optimizers.Adam(
            d_learning, beta_1=0.5) for _ in range(4)]
        
        checkpoint = tf.train.Checkpoint(generator_optimizers=self.generator_optimizers,
                                            discriminator_optimizers=self.discriminator_optimizers,
                                            generators=self.generators,
                                            discriminators=self.discriminators)

        checkpoint.restore(tf.train.latest_checkpoint(
            checkpoint_path)).expect_partial()

def generate_images(height, width, lat, dat, example_input, example_lat, example_date, example_target, generators):
    
    scales = [8, 4, 2 ,1]

    ip_images = [example_input if i == 1 else tf.image.resize(
        example_input, [height // i,  width // i]) for i in scales] 
    if(lat):
        ip_lats = [example_lat if i == 1 else tf.image.resize(
            example_lat, [height // i, width // i]) for i in scales]
    if(dat):
        ip_dates = [example_date if i == 1 else tf.image.resize(
            example_date, [height // i, width // i]) for i in scales]
    
    ip = []
    for i in range(4):
        ip.append([ip_images[i]])
        if(lat): ip[i].append(ip_lats[i])
        if(dat): ip[i].append(ip_dates[i])
    
    predictions = []
    for i in range(4):
        if(i > 0):
            ip[i].append(tf.image.resize(predictions[i-1], [predictions[i-1].shape[1]*2, predictions[i-1].shape[2]*2]))
        
        predictions.append(generators[i](ip[i], training=True))

    prediction = predictions[3] * 0.5 + 0.5
    target = example_target * 0.5 + 0.5
    plot_comparison(ip_images[3][0], target[0], prediction[0], '', save=False)

    return


def get_metrics_multi(height, width, test_dataset, generators, latitude=False, date=False):

    def sobel(img): return tf.image.sobel_edges(img)
    rmses = []
    maes = []
    mses = []
    ssims = []
    sobels = []

    for test_input, test_target, test_latitude, test_date, _ in test_dataset:
        scales = [8, 4, 2 ,1]
    
        ip_images = [test_input if i == 1 else tf.image.resize(
            test_input, [height // i, width // i]) for i in scales] 
        if(latitude):
            ip_lats = [test_latitude if i == 1 else tf.image.resize(
                test_latitude, [height // i, width // i]) for i in scales]
        if(date):
            ip_dates = [test_date if i == 1 else tf.image.resize(
                test_date, [height // i, width // i]) for i in scales]
        
        ip = []
        for i in range(4):
            ip.append([ip_images[i]])
            if(latitude): ip[i].append(ip_lats[i])
            if(date): ip[i].append(ip_dates[i])
        
        predictions = []
        for i in range(4):
            if(i > 0):
                ip[i].append(tf.image.resize(predictions[i-1], [predictions[i-1].shape[1]*2, predictions[i-1].shape[2]*2]))
            predictions.append(generators[i](ip[i], training=True))

        prediction = predictions[3] * 0.5 + 0.5
        target = test_target * 0.5 + 0.5

        mae = np.mean(np.abs(target-prediction))
        maes.append(mae)

        mse = np.mean((prediction - target) ** 2)
        mses.append(mse)

        rmse = np.sqrt(mse)

        rmses.append(rmse)

        ssim = 1 - tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(target),
                                  tf.convert_to_tensor(prediction), 1.0))
        ssims.append(ssim)

        sobel_loss = tf.reduce_mean(tf.square(
            sobel(tf.convert_to_tensor(target)) - sobel(tf.convert_to_tensor(prediction))))
        sobels.append(sobel_loss)

    return rmses, maes, mses, ssims, sobels

def test_on_image_multigan(generators, height_path, shadow_path, city, date, zoom, i, j, path=None, save=False, lat=True, dat=True):
    def predict_shadow(height_path, city, date, zoom, i, j):
        input_height, input_lat, input_date = load_input_grid(
            height_path, city, date, zoom, i, j)
        input_height, input_lat, input_date = normalize_input(
            input_height, input_lat, input_date)
        
        input_height = np.array(input_height).reshape(1, 512, 512, 1)
        input_lat = np.array(input_lat).reshape(1, 512, 512, 1)
        input_date = np.array(input_date).reshape(1, 512, 512, 1)

        scales = [8, 4, 2 ,1]
        ip_images = [input_height if i == 1 else tf.image.resize(
            input_height, [512 // i, 512 // i]) for i in scales]
        ip_lats = [input_lat if i == 1 else tf.image.resize(
            input_lat, [512 // i, 512 // i]) for i in scales]
        ip_dates = [input_date if i == 1 else tf.image.resize(
            input_date, [512 // i, 512 // i]) for i in scales]
        
        ip = []
        for i in range(4):
            ip.append([ip_images[i]])
            if(lat): ip[i].append(ip_lats[i])
            if(dat): ip[i].append(ip_dates[i])

        predictions = []
        for i in range(4):
            if(i > 0):
                ip[i].append(tf.image.resize(predictions[i-1], [predictions[i-1].shape[1]*2, predictions[i-1].shape[2]*2]))
            predictions.append(generators[i](ip[i], training=True))

        prediction = predictions[3]
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
        gt = np.where(input_height <= 0, gt, 0)
        gt = tf.cast(gt, tf.float32)
        gt = (gt / 127.5) - 1.0

        return gt

    def evaluate_shadow_prediction():

        def sobel(x): return tf.image.sobel_edges(x)

        def sobel_loss(target, gen_output): return tf.reduce_mean(
            tf.abs(sobel(target) - sobel(gen_output)))

        input_height, prediction = predict_shadow(
            height_path, city, date, zoom, i, j)
        gt = load_ground_truth(input_height, shadow_path,
                               city, date, zoom, i, j)

        prediction = prediction * 0.5 + 0.5
        gt = gt * 0.5 + 0.5

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

    evaluate_shadow_prediction()
    return