""""
Data splitting.
"""

import numpy as np


def session_split(data, boundary=0.8, validation_size=None,
                  user_id='user_id', item_id='item_id', timestamp='timestamp',
                  path_to_save=None, dataset_name=None):
    """Session-based split.

    Args:
        data (pd.DataFrame): Events data.
        boundary (float): Quantile for splitting into train and test part.
        validation_size (int): Number of users in validation set. No validation set if None.
        user_id (str): Defaults to 'user_id'.
        item_id (str): Defaults to 'item_id'.
        timestamp (str): Defaults to 'timestamp'.
        path_to_save (str): Path to save resulted data. Defaults to None.
        dataset_name (str): Name of the dataset. Defaults to None.

    Returns:
        Train, validation (optional), test datasets.
    """

    train, test = session_splitter(data, boundary, user_id, timestamp)

    if validation_size is not None:
        train, validation, test = make_validation(
            train, test, validation_size, user_id, item_id, timestamp)

        if path_to_save is not None:
            train.to_csv(path_to_save + 'train_' + dataset_name + '.csv')
            test.to_csv(path_to_save + 'test_' + dataset_name + '.csv')
            validation.to_csv(path_to_save + 'validation_' + dataset_name + '.csv')

        return train, validation, test

    if path_to_save is not None:
        train.to_csv(path_to_save + 'train_' + dataset_name + '.csv')
        test.to_csv(path_to_save + 'test_' + dataset_name + '.csv')

    else:
        train = train[[user_id, item_id, timestamp]].astype(int)
        test = test[[user_id, item_id, timestamp]].astype(int)

    return train, test

def make_validation(train, test, validation_size,
                    user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Add validation dataset."""

    validation_users = np.random.choice(train[user_id].unique(),
                                        size=validation_size, replace=False)
    validation = train[train[user_id].isin(validation_users)]
    train = train[~train[user_id].isin(validation_users)]

    train = train[[user_id, item_id, timestamp]].astype(int)
    test = test[[user_id, item_id, timestamp]].astype(int)
    validation = validation[[user_id, item_id, timestamp]].astype(int)

    return train, validation, test


def session_splitter(data, boundary, user_id='user_id', timestamp='timestamp'):
    """Make session split."""

    data.sort_values([user_id, timestamp], inplace=True)
    quant = data[timestamp].quantile(boundary)
    users_time = data.groupby(user_id)[timestamp].agg(list).apply(
        lambda x: x[1] <= quant).reset_index()
    users_time_test = data.groupby(user_id)[timestamp].agg(list).apply(
        lambda x: x[-1] > quant).reset_index()

    train_user = list(users_time[users_time[timestamp]][user_id])
    test_user = list(users_time_test[users_time_test[timestamp]][user_id])

    train = data[data[user_id].isin(train_user)]
    train = train[train[timestamp] <= quant]
    test = data[data[user_id].isin(test_user)]

    return train, test