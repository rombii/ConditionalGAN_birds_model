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

# What we expect from model:
#   - Input: Sentence
#   - Output: Image

import tensorflow_hub as hub
from Stage1 import generator as gen_s1
from Stage1 import discriminator as disc_s1
from ConditioningAugmentation import text_encoder as txt_enc
import matplotlib.pyplot as plt
import tensorflow as tf

model = gen_s1.build()

use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(use_model_url)

sentences = ["this bird has large, black, webbed feet, and is covered in gray plumage."]

embeddings = txt_enc.get_embedding(sentences)

print(embeddings)

noise = tf.random.normal([1, 512])

emb_noise = embeddings + noise

generated_images = model.predict(emb_noise)

disc_model = disc_s1.build()

disc_model.summary()

decision = disc_model.predict([generated_images, embeddings])

print(decision)

# Visualize the generated images
plt.imshow(generated_images[0])
plt.axis('off')
plt.show()




















