import tensorflow as tf


def up_sampling_block(inputs, filters):
    # Nearest-neighbor upsampling
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)

    # 3x3 convolution
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)

    # Batch normalization and ReLU activation
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def down_sampling_block(inputs, filters):
    # Convolutional layer
    x = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)

    # Batch normalization and ReLU activation
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x
