import torch
#embed dim is basically the no of features we want to represent each token with
#eg cat = [0.22,0.448,0.344]
#here the embed dim is 3 ie the len(the vector)

import torch.nn as nn

class Masked_Model(nn.Module):
    def __init__(self, embed_dim, num_heads , ff_dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True )
        #soo once we have the embed dim that is behing pushed into the nn , like how images are after converting to a vector or flat


        #expansion , non linearity , contraction
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, x):



