import random

import tensorflow as tf
import glob
import os
from ConditioningAugmentation.text_encoder import get_embedding


img_dir = 'Data/CUB_200_2011_reformat/images'
text_dir = 'Data/birds/text_c10'

# Get all files from directories
img_paths = glob.glob(os.path.join(img_dir, '**/*.jpg'), recursive=True)
text_paths = [os.path.join(text_dir, os.path.relpath(path, img_dir)).replace(".jpg", ".txt") for path in img_paths]


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = (img / 127.5) - 1
    return img


def load_text(text_path):
    text = tf.io.read_file(text_path)
    text = tf.strings.split(text, '\n')
    text = text[:-1]
    text = get_embedding(text)
    return text


def preprocess_function(img_path, text_path):
    img = load_image(img_path)
    text = load_text(text_path)
    return img, text


def load_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, text_paths))
    dataset = dataset.map(preprocess_function)

    dataset = dataset.shuffle(buffer_size=len(dataset))
    dataset = dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
