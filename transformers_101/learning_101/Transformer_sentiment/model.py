import torch.nn as nn
import math

from Deep_learning.transformers_101.learning_101.tiny_transformer.model import PositionalEncoding, TransformerBlock

class TransformerClassifier(nn.Module):
    def __init__(self, ff_dim=128, num_heads =4, embed_dim = 64, vocab_size = None  ,num_classes=2 ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.posen = PositionalEncoding(embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.posen(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)



