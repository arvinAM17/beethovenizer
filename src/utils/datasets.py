import torch
from torch.utils.data import Dataset, DataLoader

class ToySequenceDataset(Dataset):
    def __init__(self, sequences, sos_idx, eos_idx):
        """
        Args:
            sequences (List[List[int]]): List of tokenized sequences.
            sos_idx (int): Start of sequence index.
            eos_idx (int): End of sequence index.
        """
        self.sequences = []
        for seq in sequences:
            self.sequences.append([sos_idx] + seq + [eos_idx])
        self.lengths = [len(seq) for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        return seq[:-1], seq[1:]
    