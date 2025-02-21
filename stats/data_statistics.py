"""Compute data statistics."""

import pandas as pd
import numpy as np


def statistics(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Compute data statistics."""
    stats = pd.Series()
    stats['all_events'] = data.shape[0]
    stats['all_user'] = data[user_id].nunique()
    stats['all_items'] = data[item_id].nunique()
    stats['mean_length'] = np.mean(data.groupby(user_id).count()[timestamp])
    stats['density'] = stats['all_events'] / (stats['all_items'] * stats['all_user'])
    
    nunique_entries = data[[user_id, item_id]].groupby([user_id]).agg(lambda x: len(set(x))).values.flatten()
    stats['diff len'] = (stats['all_events'] - np.sum(nunique_entries)) / stats['all_events']
    
    return stats
