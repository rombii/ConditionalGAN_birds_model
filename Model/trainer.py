import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 64
noise_dim = 100  # Dimensionality of the noise for the generator


def generator_loss(fake_output):
    return tf.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output, wrong_output):
    real_loss = tf.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)(tf.ones_like(real_output), real_output)
    fake_loss = tf.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)(tf.zeros_like(fake_output), fake_output)
    wrong_loss = tf.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)(tf.zeros_like(wrong_output), wrong_output)
    total_loss = 0.5 * tf.add(real_loss, 0.5 * tf.add(fake_loss, wrong_loss))
    return total_loss


@tf.function
def train_step(images, text_embeddings, wrong_captions,
               generator, discriminator,
               gen_optimizer, disc_optimizer):
    indices = tf.random.uniform(shape=[text_embeddings.shape[0]], minval=0, maxval=10, dtype=tf.int32)
    text_embeddings = tf.gather(text_embeddings, indices, batch_dims=1)

    wrong_indices = tf.random.uniform(shape=[text_embeddings.shape[0]], minval=0, maxval=10, dtype=tf.int32)
    wrong_captions = tf.gather(wrong_captions, wrong_indices, batch_dims=1)

    noise = tf.random.normal([text_embeddings.shape[0], noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator([text_embeddings, noise], training=False)

        real_output = discriminator([images, text_embeddings], training=True)
        fake_output = discriminator([generated_images, text_embeddings], training=True)
        wrong_output = discriminator([generated_images, wrong_captions], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, wrong_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return generated_images, gen_loss, disc_loss


def train(dataset, epochs, generator, discriminator, gen_optimizer, disc_optimizer):
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        i = 0
        for image_batch, text_batch in dataset:
            wrong_text_batch = tf.random.shuffle(text_batch)
            i += 1
            generated_images, gen_loss, disc_loss = train_step(image_batch, text_batch, wrong_text_batch,
                                                               generator, discriminator,
                                                               gen_optimizer, disc_optimizer)
            print(f'Generator loss: {gen_loss} Discriminator loss: {disc_loss} Epoch: {epoch + 1} Batch: {i}')
            # Plotting and saving the generated images

        if epoch % 5 == 0:
            plt.figure(figsize=(20, 10))

            for j in range(5):
                # Rescale the generated image
                rescaled_generated_image = (generated_images[j] + 1) / 2
                rescaled_real_image = (image_batch[j] + 1) / 2

                plt.subplot(5, 2, 2*j+1)
                plt.imshow(rescaled_generated_image, interpolation='nearest')
                plt.axis('off')

                plt.subplot(5, 2, 2*j+2)
                plt.imshow(rescaled_real_image, interpolation='nearest')
                plt.axis('off')
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            try:
                plt.savefig(f'generated_images/image_epoch{epoch+1}.png')
            except OSError:
                print("Error saving image")
            plt.close()
            print('---> Saving model after epoch: ', epoch + 1)
            generator.save(f'./generators/generator_epoch{epoch+1}.h5')
            discriminator.save(f'./discriminators/discriminator_epoch{epoch+1}.h5')
