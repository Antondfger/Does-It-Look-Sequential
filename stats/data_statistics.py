"""Compute data statistics."""

import pandas as pd
import numpy as np


def statistics(data, raw_data, train, test, config, user_id='user_id',
               item_id='item_id', timestamp='timestamp'):
    """Compute data statistics."""

    stats = pd.Series()
    timecoding = config.datasets_info.time_coding
    test_time = data.timestamp.quantile(config.splitter.split_params.boundary)

    stats['full_days'] = pd.Timedelta(data.timestamp.max()- data.timestamp.min(), timecoding).days
    stats['test_days'] = pd.Timedelta(data.timestamp.max()- test_time, timecoding).days

    stats['train_events'] = train.shape[0]
    stats['test_events'] = test.shape[0]
    stats['all_events'] = data.shape[0]
    stats['all_user'] = data[user_id].nunique()
    stats['train_user'] = train[user_id].nunique()
    stats['test_user'] = test[user_id].nunique()
    stats['all_items'] = test[item_id].nunique()
    stats['mean_lenght'] = np.mean(data.groupby(user_id).count()[timestamp])
    stats['density'] = stats['all_events'] / (stats['all_items'] * stats['all_user'])


    stats['mean_train_lenght'] = np.mean(train.groupby(user_id).count()[timestamp])
    stats['med_train_lenght'] = np.median(train.groupby(user_id).count()[timestamp])
    stats['mean_test_lenght'] = np.mean(test.groupby(user_id).count()[timestamp])
    stats['med_test_lenght'] = np.median(test.groupby(user_id).count()[timestamp])

    stats['mean_timedelta_train'] = pd.Timedelta(np.mean(
        train.groupby(user_id)[timestamp].agg(lambda x: x.max() - x.min())), timecoding).days

    stats['med_timedelta_train'] = pd.Timedelta(np.median(
        train.groupby(user_id)[timestamp].agg(lambda x: x.max() - x.min())), timecoding).days

    stats['mean_timedelta_test'] = pd.Timedelta(np.mean(
        test.groupby(user_id)[timestamp].agg(lambda x: x.max() - x.min())), timecoding).days

    stats['med_timedelta_test'] = pd.Timedelta(np.median(
        test.groupby(user_id)[timestamp].agg(lambda x: x.max() - x.min())), timecoding).days

    stats['events_without_duplicates_timestamp'] = \
        raw_data[[user_id, timestamp]].drop_duplicates().shape[0]
    stats['normal_events_without_duplicates_timestamp'] = \
        stats['events_without_duplicates_timestamp'] / stats['all_events']

    count = data.groupby([user_id, item_id]).size().reset_index(name='count')
    count = count[count['count'] > 1]
    num_users_with_duplicates = count[user_id].nunique()

    stats['num_users_with_duplicates_item_in_sequence'] = num_users_with_duplicates
    stats['percent_user_with_duplicates'] = num_users_with_duplicates / stats['all_user']

    user_seq = np.array(data.groupby(user_id)[item_id].size())
    user_seq_without_repeats = np.array(data.groupby(user_id)[item_id].nunique())

    stats['mean_diff_len_and_unique_len'] = np.mean(user_seq - user_seq_without_repeats)
    stats['normal_diff_len_and_unique_len'] = \
        stats['mean_diff_len_and_unique_len'] / np.mean(user_seq)

    raw_data.sort_values([user_id, timestamp], inplace=True)
    raw_data['user_item'] = raw_data[user_id].astype(str) + '_' + raw_data[item_id].astype(str)

    stats['number_of_i-i'] = (raw_data['user_item'].shift() == raw_data['user_item']).sum()
    stats['normal_number_of_i-i'] = stats['number_of_i-i'] / raw_data.shape[0]

    return stats
