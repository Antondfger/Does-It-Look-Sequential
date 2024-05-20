import numpy as np


def shuffle(data, seed=17, user_id='user_id', item_id='item_id'):
    """Shuffle user sequences."""

    np.random.seed(seed)
    shuffled_data = data.copy()
    shuffled_data[item_id] = data.groupby(user_id)[item_id].transform(np.random.permutation)
    shuffled_data.reset_index(drop=True, inplace=True)

    return shuffled_data


def remove_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Remove last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    short_data = data.groupby(user_id)[item_id].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(item_id)
    short_data[timestamp] = data.groupby(user_id)[timestamp].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(timestamp)[timestamp]

    return short_data


def get_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Get last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    data_last = data.groupby(user_id)[item_id].agg(list).apply(lambda x: x[-1]).reset_index()

    return data_last
