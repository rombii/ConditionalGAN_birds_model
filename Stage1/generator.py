import tensorflow as tf
from Helper.blocks import up_sampling_block


def build():
    input_size = 512  # Size of the input noise vector
    filters = 64

    # Input layer
    inputs = tf.keras.layers.Input(shape=(input_size,))

    # Reshape the input to prepare for convolution
    reshaped = tf.keras.layers.Reshape((1, 1, input_size))(inputs)

    reshaped = tf.keras.layers.BatchNormalization()(reshaped)
    reshaped = tf.keras.layers.ReLU()(reshaped)

    # Repeat upsampling block 6 times (64x64x3)
    for _ in range(6):
        reshaped = up_sampling_block(reshaped, filters)
        filters //= 2  # Reduce the number of filters with each block

    # Final convolution to get a 64x64 image
    final_output = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(
        reshaped)

    # Define the generator model
    model = tf.keras.Model(inputs=inputs, outputs=final_output)
    return model
