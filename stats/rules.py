"""
Count sequential rules.
"""

from collections import Counter

import pandas as pd
from nltk.util import ngrams

from preprocessing.preparation import shuffle as make_shuffle


def rule_counter(data, n_gram_order=2, support_threshold=5, confidence_threshold=0.01,
                 user_id='user_id', item_id='item_id', timestamp='timestamp', random_state=17):
    """Count sequential rules.
    
    Args:
        data (pd.DataFrame): Events data.
        n_gram_order (int): N-grams order to compute.
        support_threshold (int): Threshold for the occurrence of n-grams in data.
        confidence_theshold (float): Threshold for confidence of n-grams in data.
        user_id (str): Defaults to 'user_id'.
        item_id (str): Defaults to 'item_id'.
        timestamp (str): Defaults to 'timestamp'.
       
    Returns:
        pd.Series with calculated statistics.
    """

    statistics = pd.Series()

    data.sort_values([user_id, timestamp], inplace=True)
    data_seq = list(data.groupby([user_id])[item_id].agg(list))

    shuffle = make_shuffle(data, seed=random_state)

    shuffle.sort_values([user_id, timestamp], inplace=True)
    shuffle_seq = list(shuffle.groupby([user_id])[item_id].agg(list))

    data_counter = count_n_grams(data_seq, n_gram_order)
    shuffle_counter = count_n_grams(shuffle_seq, n_gram_order)

    statistics['all_rule'] = len(shuffle_counter) / len(data_counter)
    statistics['all_rule_data'] = len(data_counter)
    statistics['all_rule_shuffle'] = len(shuffle_counter)

    data_counter_support = {key: value for key, value in data_counter.items()
                           if value > support_threshold}
    shuffle_counter_support = {key: value for key, value in shuffle_counter.items()
                              if value > support_threshold}
    statistics['support_rule'] = len(shuffle_counter_support) / len(data_counter_support)
    statistics['support_rule_data'] = len(data_counter_support)
    statistics['support_rule_shuffle'] = len(shuffle_counter_support)

    count_item_orginal = count_n_grams(data_seq, n_gram_order - 1)
    count_item_shuffle = count_n_grams(shuffle_seq, n_gram_order - 1)

    data_stat_conf = {key: value/count_item_orginal[key[:-1]]
                      for key, value in data_counter.items()
                      if value/count_item_orginal[key[:-1]] > confidence_threshold}
    shuffle_stat_conf = {key: value/count_item_shuffle[key[:-1]]
                         for key, value in shuffle_counter.items()
                         if value/count_item_shuffle[key[:-1]] > confidence_threshold}
    statistics['confidence_rule'] = len(shuffle_stat_conf) / len(data_stat_conf)
    statistics['confidence_rule_data'] = len(data_stat_conf)
    statistics['confidence_rule_shuffle'] = len(shuffle_stat_conf)

    data_stat_conf = {key: value/count_item_orginal[key[:-1]]
                      for key, value in data_counter_support.items()
                      if value/count_item_orginal[key[:-1]] > confidence_threshold}
    shuffle_stat_conf = {key: value/count_item_shuffle[key[:-1]]
                         for key, value in shuffle_counter_support.items()
                         if value/count_item_shuffle[key[:-1]] > confidence_threshold}
    statistics['confidence_and_support_rule'] = len(shuffle_stat_conf) / len(data_stat_conf)
    statistics['confidence_and_support_data'] = len(data_stat_conf)
    statistics['confidence_and_support_shuffle'] = len(shuffle_stat_conf)

    return statistics


def count_n_grams(sequence, n_gram_order):
    """Count n-grams of given order."""

    counter = Counter()
    for seq in sequence:
        ngrams_list = list(ngrams(seq, n_gram_order))
        counter.update(ngrams_list)

    return counter
