import tensorflow as tf


def get_embedding(sentences):
    # Load the Universal Sentence Encoder model
    use_model_url = 'ConditioningAugmentation'
    embed = tf.saved_model.load(use_model_url)

    # Get embeddings for the sentences
    embeddings = embed(sentences)

    # Return the embeddings without reshaping
    return embeddings


def conditioning_augmentation(x):
    limit = x.shape[1] // 2
    mean = x[:, :limit]
    log_sigma = x[:, limit:]

    stddev = tf.math.exp(log_sigma)
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean.shape[1], ), dtype='float')
    c = mean + stddev * epsilon
    return c

