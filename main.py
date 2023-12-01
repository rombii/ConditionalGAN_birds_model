# Structure of StackGAN:
#   - Conditioning Augmentation
#         - Input Sentence
#         - Text Encoder
#         - Conditional Vectorizer
#   - Stage-I GAN: text-to-image mapping
#         - Generator
#                - Input Conditional Vector and noise
#                - Upsampling from vector to image
#                - Output Image
#         - Discriminator
#                - Input Image, Text Embedding
#                - Downsampling from image to vector
#                - Downsampling from text embedding to 3d vector
#                - Concatenate the two vectors
#                - Output Decision
#   - Stage-II GAN: image refinement
#        - TBA
import glob
import os

# What we expect from model:
#   - Input: Sentence
#   - Output: Image

import tensorflow_hub as hub
from Stage1 import generator as gen_s1
from Stage1 import discriminator as disc_s1
from Stage1 import trainer as train_s1
from ConditioningAugmentation import text_encoder as txt_enc
from Data import loader as data_loader
import matplotlib.pyplot as plt
import tensorflow as tf


data = data_loader.load_dataset()

gen_model = gen_s1.build()

disc_model = disc_s1.build()

train_s1.train(data, 600,
               gen_model, disc_model,
               tf.keras.optimizers.Adam(2e-4), tf.keras.optimizers.Adam(2e-4))




















