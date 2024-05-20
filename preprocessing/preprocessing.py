"""
Data preprocessing.
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np


def preprocessing(data, users_sample=None, item_min_count=5, min_len=5, core=True, encoding=True,
                  drop_repeats=False, user_id='user_id', item_id='item_id', timestamp='timestamp',
                  path_to_save=None):
    """Data preprocessing.

    Args:
        data (pd.DataFrame): Events data.
        users_sample (int): Number of users for sampling. Defaults to None.
        item_min_count (int): Minimum occurrence of items. Defaults to 5.
        min_len (int): Minimum sequence length. Defaults to 5. 
        core (boolean): Make core or usual user/item filtering. Defaults to True. 
        encoding (boolean): Make user/item encoding or not. Defaults to True. 
        drop_repeats (boolean): Remove repeated items like i-i-j->i-j or not. Defaults to True. 
        user_id (str): Defaults to 'user_id'.
        item_id (str): Defaults to 'item_id'.
        timestamp (str): Defaults to 'timestamp'.
        path_to_save (str): Path to save preprocessed data. Defaults to None.
 
    Returns:
        Data with filtered items by occurrence, sequences by length, and deleted repeats.
    """

    data = rename(data, user_id, item_id, timestamp)

    print_stats(data, text='raw data')

    if users_sample is not None:

        sampled_users = np.random.choice(data['user_id'].unique(),
                                         size=users_sample, replace=False)
        data = data[data['user_id'].isin(sampled_users)]
        print_stats(data, text='after user sampling')

    if core:

        step = 1
        while (data['user_id'].value_counts().min() < min_len
               or data['item_id'].value_counts().min() < item_min_count):

            print(f'n-core filtering step {step}')
            data = drop_short_sequences(data, min_len)
            data = filter_items(data, item_min_count)
            if drop_repeats:
                data = drop_repeated_items(data)
            step += 1

    else:

        data = drop_short_sequences(data, min_len)
        data = filter_items(data, item_min_count)
        if drop_repeats:
            data = drop_repeated_items(data)

    if encoding:
        data = encode_items(data)
        data = encode_users(data)

    if path_to_save is not None:
        data.to_csv(path_to_save)

    return data


def rename(data, user_id="user_id", item_id='item_id', timestamp='timestamp'):
    "Rename columns of dataframe"

    columns_rename = {user_id:'user_id',
                      item_id:'item_id',
                      timestamp:'timestamp'}
    data = data.rename(columns=columns_rename)

    return data


def filter_items(data, item_min_count, item_id="item_id"):
    """Filter items by occurrence threshold."""

    counts = data[item_id].value_counts()
    data = data[data[item_id].isin(counts[counts >= item_min_count].index)]

    print_stats(data, text='after item filtering', extended=True)

    return data


def drop_repeated_items(data, user_id='user_id', item_id="item_id", timestamp="timestamp"):
    """Remove repeated items like i-i-j -> i-j."""

    data.sort_values([user_id, timestamp], inplace=True)
    data['user_item'] = data[user_id].astype(str) + '_' + data[item_id].astype(str)

    while (data['user_item'].shift() == data['user_item']).sum() != 0:
        not_duplicates_ind = data['user_item'].shift() != data['user_item']
        data = data.loc[not_duplicates_ind]

    data = data.drop('user_item', axis=1)
    print_stats(data, text='after dropping repeated items', extended=True)

    return data


def drop_short_sequences(data, min_len, user_id='user_id'):
    """Drop user sequences shorter than given threshold."""

    counts = data[user_id].value_counts()
    users = counts[counts >= min_len].index
    data = data[data[user_id].isin(users)]

    print_stats(data, text='after dropping short sequences', extended=True)

    return data


def encode_items(data, item_id='item_id'):
    """Encode items to consecutive ids."""

    encoder = LabelEncoder()
    data[item_id] = encoder.fit_transform(data[item_id])

    return data


def encode_users(data, user_id='user_id'):
    """Encode items to consecutive ids."""

    encoder = LabelEncoder()
    data[user_id] = encoder.fit_transform(data[user_id])

    return data


def print_stats(data, text=None, extended=False):
    """Print data statistics."""

    if text is not None:
        text = text + ': '
    else:
        text = ''

    text = (text + f'interactions {len(data)} '
            + f'users {data.user_id.nunique()} '
            + f'items {data.item_id.nunique()} ')

    if extended:
        text = (text + f'min_seq_len {data.user_id.value_counts().min()} '
                f'min_item_occurence {data.item_id.value_counts().min()}')

    print(text)
