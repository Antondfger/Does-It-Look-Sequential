"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """
    Base dataset class.
    """

    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=False,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):

        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):

        return len(self.data)

    def sample_negatives(self, item_sequence):
        """Sample negative items for loss calculation."""

        negatives = self.all_items[~np.isin(self.all_items, item_sequence)]
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            # replace=True for speed, with replace=False sampling can be very slow
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=True)

        return negatives


class CausalLMDataset(LMDataset):
    """Dataset for training."""

    def __init__(self, df, max_length=128,
                 shift_labels=True, num_negatives=None,
                 full_negative_sampling=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.shift_labels = shift_labels

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]
        if len(item_sequence) > self.max_length + 1:
            item_sequence = item_sequence[-self.max_length - 1:]

        input_ids = np.array(item_sequence[:-1])
        if self.shift_labels:
            labels = np.array(item_sequence[1:])
        else:
            labels = input_ids

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class CausalLMPredictionDataset(LMDataset):
    """Dataset for validation and prediction."""

    def __init__(self, df, max_length=128, validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length-1:-1]
            item_sequence = item_sequence[:-1]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            input_ids = item_sequence[-self.max_length:]

            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}


class PaddingCollateFn:
    """Collate function for proper padding."""

    def __init__(self, padding_value=0, labels_padding_value=-100, left_padding=False):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value
        self.left_padding = left_padding

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key == 'labels':
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value
            # left padding is required for sequence generation with huggingface models
            if self.left_padding:
                values = [torch.tensor(example[key][::-1].copy()) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value).flip(-1)
            else:
                values = [torch.tensor(example[key]) for example in batch]
                collated_batch[key] = pad_sequence(values, batch_first=True,
                                                   padding_value=padding_value)
        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch
