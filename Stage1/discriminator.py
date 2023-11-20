import tensorflow as tf
from Helper.blocks import down_sampling_block


def build():
    Nd = 3  # Nd dimensions
    Md = 64  # Md dimensions

    # Input layers
    text_embedding = tf.keras.layers.Input(shape=(512,))
    image_input = tf.keras.layers.Input(shape=(Md, Md, 3))

    # Compress text embedding
    compressed_text = tf.keras.layers.Dense(Nd, use_bias=False)(text_embedding)

    # Spatially replicate to form Md x Md x Nd tensor
    replicated_text = tf.tile(tf.expand_dims(tf.expand_dims(compressed_text, 1), 1), [1, 4, 4, 512])

    # Down-sampling blocks for the image
    x = image_input
    x = down_sampling_block(x, filters=64)
    x = down_sampling_block(x, filters=128)
    x = down_sampling_block(x, filters=256)
    x = down_sampling_block(x, filters=512)

    # Concatenate image and text tensors along the channel dimension
    concatenated_tensor = tf.keras.layers.Concatenate(axis=-1)([x, replicated_text])

    # 1x1 convolutional layer
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(
        concatenated_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # Flatten the tensor
    x = tf.keras.layers.Flatten()(x)

    # Fully connected layer with one node
    output = tf.keras.layers.Dense(1)(x)

    # Define the discriminator model
    model = tf.keras.models.Model(inputs=[image_input, text_embedding], outputs=output)

    return model