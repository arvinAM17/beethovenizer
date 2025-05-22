from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset


def sort_and_batch(dataset, batch_size):
    """
    Sorts dataset by sequence length and returns batches of indices with similar lengths.
    """
    lengths = dataset.lengths  # assumes (input, target) pairs
    sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])

    batches = []
    for i in range(0, len(sorted_indices), batch_size):
        batch = sorted_indices[i:i + batch_size]
        batches.append(batch)

    return batches


def custom_collate_fn(pad_idx):
    def collate(batch):
        inputs, targets = zip(*batch)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
        targets = pad_sequence(targets, batch_first=True,
                               padding_value=pad_idx)
        return inputs, targets
    return collate
