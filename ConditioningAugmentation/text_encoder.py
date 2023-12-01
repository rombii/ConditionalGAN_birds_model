import tensorflow_hub as hub
import tensorflow as tf


def get_embedding(sentences):
    # Load the Universal Sentence Encoder model
    use_model_url = 'ConditioningAugmentation'
    embed = hub.load(use_model_url)

    # Get embeddings for the sentences
    embeddings = embed(sentences)

    # Return the embeddings without reshaping
    return embeddings


