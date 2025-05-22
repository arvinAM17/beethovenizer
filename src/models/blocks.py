import torch.nn as nn


class EmbeddingBlock(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 300, padding_idx=None):
        """
        Embedding block for the model.
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding.
            padding_idx (int, optional): Padding index for the embedding. Defaults to None.
        """
        super(EmbeddingBlock, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        out = self.embedding(x)
        return out


class LSTMBlock(nn.Module):
    """
    LSTM block for the model.
    Args:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Defaults to True.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=batch_first)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, (h, c)


class FCBlock(nn.Module):
    """
    Fully connected block for the model.
    Args:
        input_size (int): Size of the input.
        output_size (int): Size of the output.
        activation (callable, optional): Activation function. Defaults to None.
    """

    def __init__(self, input_size, output_size, activation=None):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        out = self.fc(x)
        if self.activation:
            out = self.activation(out)
        return out
