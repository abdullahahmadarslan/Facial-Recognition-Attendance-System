import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec1, vec2: numpy.ndarray
        Input vectors.

    Returns:
    float
        Cosine similarity value.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # Avoid division by zero
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def match_embedding(new_embedding, db_embeddings, threshold=0.9):
    """
    Compare the new embedding against database embeddings using cosine similarity,
    and return the tuple with the highest similarity that exceeds the threshold.

    Parameters:
    new_embedding: numpy.ndarray
        The real-time embedding to match.
    db_embeddings: list of numpy.ndarray
        List of embeddings from the database.
    threshold: float
        The threshold for cosine similarity to consider a match.

    Returns:
    list
        A list containing the tuple with the index and similarity of the best match.
    """
    best_match = None
    max_similarity = -1  # Initialize with a very low similarity to ensure any match is larger

    # Iterate through each embedding in the database and calculate similarity
    for i, db_embedding in enumerate(db_embeddings):
        similarity = cosine_similarity(new_embedding, db_embedding)
        if similarity >= threshold and similarity > max_similarity:
            best_match = (i, similarity)  # Update the best match if the similarity is higher
            max_similarity = similarity  # Update the max similarity value

    # If a match was found, return the best match as a list with one tuple
    if best_match:
        return [best_match]  # Return the best match in a list
    else:
        return []  # If no match found, return an empty list
