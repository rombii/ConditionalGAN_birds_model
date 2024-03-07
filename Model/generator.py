import tensorflow as tf
from ConditioningAugmentation import text_encoder


def build():
    text_input = tf.keras.layers.Input(shape=(512,))
    noise_input = tf.keras.layers.Input(shape=(100,))

    x = tf.keras.layers.Dense(256)(text_input)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Lambda(text_encoder.conditioning_augmentation)(x)

    gen_input = tf.keras.layers.Concatenate(axis=1)([x, noise_input])

    x = tf.keras.layers.Dense(4*4*1024, use_bias=False)(gen_input)
    x = tf.keras.layers.Activation('leaky_relu')(x)
    x = tf.keras.layers.Reshape((4, 4, 1024))(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Activation('tanh')(x)

    model = tf.keras.Model(inputs=[text_input, noise_input], outputs=x)

    return model

