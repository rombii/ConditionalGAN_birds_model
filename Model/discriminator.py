import tensorflow as tf


def build():
    image_input = tf.keras.layers.Input(shape=(64, 64, 3))
    text_input = tf.keras.layers.Input(shape=(512,))

    downsampled_image = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_uniform')(image_input)
    downsampled_image = tf.keras.layers.LeakyReLU(alpha=0.2)(downsampled_image)

    downsampled_image = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_uniform')(downsampled_image)
    downsampled_image = tf.keras.layers.BatchNormalization()(downsampled_image)
    downsampled_image = tf.keras.layers.LeakyReLU(alpha=0.2)(downsampled_image)

    downsampled_image = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_uniform')(downsampled_image)
    downsampled_image = tf.keras.layers.BatchNormalization()(downsampled_image)
    downsampled_image = tf.keras.layers.LeakyReLU(alpha=0.2)(downsampled_image)

    downsampled_image = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), use_bias=False, padding='same', kernel_initializer='he_uniform')(downsampled_image)
    downsampled_image = tf.keras.layers.BatchNormalization()(downsampled_image)
    downsampled_image = tf.keras.layers.LeakyReLU(alpha=0.2)(downsampled_image)

    transformed_text = tf.keras.layers.Dense(128)(text_input)
    transformed_text = tf.keras.layers.ReLU()(transformed_text)
    transformed_text = tf.keras.layers.Reshape((1, 1, 128))(transformed_text)
    transformed_text = tf.keras.layers.UpSampling2D(size=(4, 4))(transformed_text)

    concatenated = tf.keras.layers.concatenate([downsampled_image, transformed_text])

    x = tf.keras.layers.Conv2D(512, (1, 1), use_bias=False, padding='same', kernel_initializer='he_uniform')(concatenated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[image_input, text_input], outputs=x)

    return model
