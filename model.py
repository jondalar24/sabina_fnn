import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramLanguageModeler(nn.Module):
    """
    Modelo de lenguaje basado en N-gramas que predice la siguiente palabra dados 'context_size' tokens previos.
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()

        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(inputs)  # (batch_size, context_size, embedding_dim)
        embeds = torch.reshape(embeds, (-1, self.context_size * self.embedding_dim))  # (batch_size, context_size * embedding_dim)
        out = F.relu(self.linear1(embeds))  # (batch_size, 128)
        out = self.linear2(out)  # (batch_size, vocab_size)
        return out
