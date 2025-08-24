import torch
from torch.utils.data import Dataset

text = "to be or not to be is the real question"

#creating a sorted list of unique chars
chars = sorted(list(set(text)))

vocab_size  =  len(chars)

#chars to idx
stoi = {ch : i for i , ch in enumerate(chars)}

#idx to chars
itos =  {i : ch for ch,  i in stoi.items()}

print(stoi)
print(itos)

def encode(s):
    return [stoi[c] for c in s]
    #basically turns a string to the numeric code by lookup

def decode(l):
    return "".join(itos[i] for i in l)
    #turns numeric code back to string after lookup


#converting the text into a tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data)

class CharDataset(Dataset):
    def __init__(self, data , block_size):
        self.data = data
        self.block_size = block_size
        #basic initialization

    def __len__(self):
        return len(self.data) - self.block_size
        #basically this will tell how many samples can be created
        # len(data)list - block size  , tells us how many valid possible sample we can create without
        #going out of bounds , (x,y)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx+1 : idx+self.block_size+1]
        return x , y
        #basically gives back the sample or sequences within range , connected to len
        #the current samples and the next samples