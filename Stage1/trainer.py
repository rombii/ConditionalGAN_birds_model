import os
import pickle

import tensorflow as tf
import matplotlib.pyplot as plt


BATCH_SIZE = 64
noise_dim = 512  # Dimensionality of the noise for the generator


def generator_loss(fake_output):
    return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output, wrong_output):
    real_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    wrong_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(wrong_output), wrong_output)
    total_loss = real_loss + fake_loss + wrong_loss
    return total_loss


@tf.function
def train_step(images, text_embeddings, wrong_captions, generator, discriminator, gen_optimizer, disc_optimizer):
    text_embeddings = tf.reduce_mean(text_embeddings, axis=1)
    text_embeddings_noise = tf.random.normal([text_embeddings.shape[0], 512])
    text_embeddings = tf.concat([text_embeddings, text_embeddings_noise], axis=1)

    wrong_captions = tf.reduce_mean(wrong_captions, axis=1)
    wrong_captions_noise = tf.random.normal([wrong_captions.shape[0], 512])
    wrong_captions = tf.concat([wrong_captions, wrong_captions_noise], axis=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(text_embeddings, training=True)

        real_output = discriminator([images, text_embeddings], training=True)
        fake_output = discriminator([generated_images, text_embeddings], training=True)
        wrong_output = discriminator([images, wrong_captions], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, wrong_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    # Returns the generator and discriminator losses for possible debugging/monitoring
    return generated_images, gen_loss, disc_loss


def train(dataset, epochs, generator, discriminator, gen_optimizer, disc_optimizer):

    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Filename to save weights
    save_file_gen = os.path.join(save_dir, 'gen_model_weights.h5')
    save_file_disc = os.path.join(save_dir, 'disc_model_weights.h5')

    if os.path.isfile(save_file_gen) and os.path.isfile(save_file_disc):
        print("=> Loading checkpoint...")
        generator.load_weights(save_file_gen)
        discriminator.load_weights(save_file_disc)
        print("Checkpoint loaded")
    else:
        print("No checkpoint found")
    epoch_file = os.path.join(save_dir, 'epoch.pkl')
    if os.path.exists(epoch_file):
        with open(epoch_file, 'rb') as f:
            start_epoch = pickle.load(f)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch + 1}')
        i = 0
        for image_batch, text_batch in dataset:  # Change here, use train_dataset
            i += 1
            wrong_text_batch = tf.random.shuffle(text_batch)
            generated_images, gen_loss, disc_loss = train_step(image_batch, text_batch, wrong_text_batch,
                                                               generator, discriminator, gen_optimizer, disc_optimizer)
            print(f'Generator loss: {gen_loss} Discriminator loss: {disc_loss} Epoch: {epoch + 1} Batch: {i}')
            # Plotting and saving the generated images outside the tf.Function
            plt.imshow(generated_images[0])
            plt.axis('off')
            plt.savefig(f'generated_images/image_epoch{epoch}.png')  # save the image to file
            plt.close()
        print('---> Saving model after epoch: ', epoch)
        generator.save_weights(save_file_gen)
        discriminator.save_weights(save_file_disc)
        with open(epoch_file, 'wb') as f:
            pickle.dump(epoch + 1, f)

        if epoch % 100 == 0:
            gen_optimizer.learning_rate = gen_optimizer.learning_rate * 0.5
            disc_optimizer.learning_rate = disc_optimizer.learning_rate * 0.5
