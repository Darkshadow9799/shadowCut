import tensorflow as tf

class Pre_GAN:
    def __init__(self, HEIGHT, WIDTH, CHANNEL, STRIDES = 2, KERNEL_SIZE = 5, 
                FILTER_ARRAY_SIZE = 5, FILTERS = 32 ,OUTPUT_STRIDES = 32, ALPHA = 0.2,
                LATENT_DIM = 100, W_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                B_INIT = tf.keras.initializers.Zeros()):
        self.H = HEIGHT
        self.W = WIDTH
        self.C = CHANNEL
        self.W_INIT = W_INIT
        self.B_INIT = B_INIT
        self.STRIDES = STRIDES
        self.KERNEL_SIZE = KERNEL_SIZE
        self.FILTER_ARRAY_SIZE = FILTER_ARRAY_SIZE
        self.OUTPUT_STRIDES = OUTPUT_STRIDES
        self.ALPHA = ALPHA
        self.LATENT_DIM = LATENT_DIM
        self.FILTERS = FILTERS
    
    '''
    CONVOLUTIONAL BLOCK
    '''
    def conv_block(self, inputs, num_filters, padding="same", activation=True):
        x = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size= self.KERNEL_SIZE,
            kernel_initializer=self.W_INIT,
            bias_initializer=self.B_INIT, 
            padding=padding,
            strides= self.STRIDES,
        )(inputs)

        if activation:
            x = tf.keras.layers.LeakyReLU(alpha=self.ALPHA)(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        return x

    '''
    DECONVOLUTIONAL BLOCK
    '''
    def deconv_block(self, inputs, num_filters, padding="same", bn=True):
        x = tf.keras.layers.Conv2DTranspose(
            filters=num_filters,
            kernel_size= self.KERNEL_SIZE,
            kernel_initializer=self.W_INIT,
            bias_initializer=self.B_INIT, 
            padding=padding,
            strides= self.STRIDES,
            use_bias=False
            )(inputs)

        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(self.ALPHA)(x)
        return x

    '''
    BUILD GENERATOR
    '''
    def build_generator(self):
        f = [2**i for i in range(self.FILTER_ARRAY_SIZE)][::-1]
        filters = self.FILTERS
        output_strides = self.OUTPUT_STRIDES
        h_output = self.H // output_strides
        w_output = self.W // output_strides
        noise = tf.keras.Input(shape=(self.LATENT_DIM,), name="generator_noise_input")

        x = tf.keras.layers.Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Reshape((h_output, w_output, 16 * filters))(x)
        

        for i in range(1, 5):
            x = self.deconv_block(x,
                num_filters=f[i] * filters,
                kernel_size= self.KERNEL_SIZE,
                strides= self.STRIDES,
                bn=True
            )

        x = self.deconv_block(x,
            num_filters=3,  ## Change this to 1 for grayscale.
            kernel_size= self.KERNEL_SIZE,
            strides= self.STRIDES,
            bn=False
        )

        fake_output = tf.keras.layers.Activation("tanh")(x)

        return tf.keras.models.Model(noise, fake_output, name="generator")

    '''
    BUILD DISCRIMINATOR
    '''
    def build_discriminator(self):
        f = [2**i for i in range(self.FILTER_ARRAY_SIZE - 1)]
        image_input = tf.keras.Input(shape=(self.H, self.W, self.C))
        x = image_input
        filters = self.FILTERS
        output_strides = self.OUTPUT_STRIDES
        h_output = self.H // output_strides
        w_output = self.W // output_strides

        for i in range(0, 4):
            x = self.conv_block(x,
                            num_filters=f[i] * filters, 
                            kernel_size= self.KERNEL_SIZE, 
                            strides= self.STRIDES)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.models.Model(image_input, x, name="discriminator")

        
''' Remaining'''
class GAN(tf.keras.models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}
