"""
Compute jaccard similarity.
"""

import numpy as np


def jaccard_similarity(data, shuffle, user_id='user_id', item_id='item_id'):
    """Jaccard similarity between shuffled and original data.
    
    Args:
        data (pd.DataFrame): Events data.
        shuffle (pd.DataFrame): Shuffle events data.
        user_id (str): Defaults to 'user_id'.
        item_id (str): Defaults to 'item_id'.
        
    Returns:
        Jacard similarity.
    """

    data_sequence = data.groupby(user_id).agg(list)[item_id]
    shuffle_sequence = shuffle.groupby(user_id).agg(list)[item_id]

    jaccard = []
    for first_seq, second_seq in zip(data_sequence, shuffle_sequence):
        jaccard.append(compute_jaccard(first_seq, second_seq))

    return np.mean(jaccard)


def compute_jaccard(first_seq, second_seq):
    """Calculates the jaccard metric for two sequnces of items."""
    first_set = set(first_seq)
    second_set = set(second_seq)
    return len(first_set.intersection(second_set)) / len(first_set.union(second_set))
