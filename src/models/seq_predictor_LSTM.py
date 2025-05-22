import torch
import torch.nn as nn
import torch.optim as optim
from .blocks import LSTMBlock, FCBlock, EmbeddingBlock
from src.training.train_seq_predictor import train_epoch, train_model


class SeqPredictorLSTM(nn.Module):
    """
    Sequence Predictor using LSTM.
    Args:
        vocab_size (int): Size of the vocabulary.
        pad_idx (int, optional): Padding index for the embedding. Defaults to 0.
        embedding_dim (int, optional): Dimension of the embedding. Defaults to 300.
        hidden_size (int, optional): Size of the hidden state. Defaults to 512.
        num_layers (int, optional): Number of LSTM layers. Defaults to 1.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, embedding_dim: int = 300, hidden_size: int = 512, num_layers=1):
        super(SeqPredictorLSTM, self).__init__()

        self.pad_idx = pad_idx

        self.embedding = EmbeddingBlock(vocab_size, embedding_dim, pad_idx)
        self.lstm = LSTMBlock(embedding_dim,
                              hidden_size, num_layers)
        self.fc = FCBlock(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        out = self.fc(out)
        return out, (h, c)

    def fit(self, dataloader, epochs: int = 10, optimizer=optim.SGD, criterion=nn.CrossEntropyLoss, device=None, writer=None):
        """
        Fit the model to the data.
        Args:
            dataloader (DataLoader): DataLoader for the training data.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            optimizer (torch.optim.Optimizer, optional): Optimizer for the model. Defaults to optim.SGD.
            criterion (torch.nn.Module, optional): Loss function. Defaults to nn.CrossEntropyLoss.
            device (torch.device, optional): Device to train on. Defaults to None.
            writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.mps.is_available() else
                                  "cpu")
            self.to(device)

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        elif isinstance(optimizer, type):
            optimizer = optimizer(self.parameters(), lr=1e-3)

        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        elif isinstance(criterion, type):
            criterion = criterion(ignore_index=self.pad_idx)

        train_model(self, dataloader, optimizer, criterion,
                    device, epochs, writer)
