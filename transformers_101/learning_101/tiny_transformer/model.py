import torch.nn as nn
import torch
import math

#basically the model doesnt know the order of the words cause it looks at it together
#we use positional encoding to give it a position like here this is token 1, this is token 2 etc
#initially each token will lead to a vector for meaning but with this the vector will contrian meaning + postion information

class PositionalEncoding(nn.Module):
    def __init__(self, d_model , max_len= 5000):
        super().__init__()
        #creating a matrix of 0 to fill later
        pe =  torch.zeros(max_len, d_model)

        #from x to xy
        position = torch.arange(0, max_len , dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) #check notes

        #applying it in all rows : and conditional columns, like sin for all even, cos for all odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        #makes it more convenient for later

        #basically saves it as non learning para, so it doesnt get updates during backprop
        self.register_buffer('pe',pe)

    def forward(self , x):
        return x + self.pe[:,:x.size(1)]
        #takes the size of the related idx mentioned [batch_size, seq_len, d_model]
        #here it basically takes the len of the seq instead of the max size

#embed dim is basically the size of the learned vector
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads , ff_dim):
        super().__init__()#embed dim is basically the size of the vector that stores the embeddings for each character
        #ff dim is basically the no of nodes in the feedforward nn hidden layer
        self.attn = nn.MultiheadAttention(embed_dim , num_heads, batch_first=True)
        #we use FNN cause the attention layer is context aware but not great at complex transformations
        #it allows more expressive feature transformations
        self.ff = nn.Sequential(
            nn.Linear(embed_dim , ff_dim), # expands dimensions
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)#projects dim back to start one
        )
        #after attention + residual connection
        self.norm1 = nn.LayerNorm(embed_dim)
        #after feedforward + residual connection
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x,x,x) # understand context
        x = self.norm1(x + attn_output) #stability for training

        ff_output = self.ff(x) #complex transformations per token
        x = self.norm2(x + ff_output) #stability

        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size , embed_dim = 32 , num_heads = 2 ,ff_dim = 64):
        super().__init__()

        self.embed = nn.Embedding(vocab_size , embed_dim)#actual word embeddings, size based on embed dim
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer = TransformerBlock(embed_dim , num_heads , ff_dim)
        #looks at word  , tries to figure which are important , then passes to nn
        self.fc_out =  nn.Linear(embed_dim , vocab_size) # here vocab size is the no of class , like in CNN, the final output
        #takes transformers results for each word , convert back to vocab scores
        #eg the output vector says this word is 90% to be dog  , 5% to be cat

    def forward(self, x):
        x = self.embed(x)
        x  = self.positional_encoding(x)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits