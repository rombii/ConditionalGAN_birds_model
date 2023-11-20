import tensorflow_hub as hub


def get_embedding(sentences):
    # Load the Universal Sentence Encoder model
    use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(use_model_url)

    # Get embeddings for the sentences
    embeddings = embed(sentences)

    # Print the embeddings
    return embeddings

