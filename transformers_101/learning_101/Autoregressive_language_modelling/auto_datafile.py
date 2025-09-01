import torch
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections import Counter
import nltk
dataset = load_dataset("roneneldan/TinyStories")
train_dataset = dataset["train"]
#test_dataset = dataset["test"]
nltk.download("punkt")

top_k = 50_000


#create vocab from training text
print("reached here")
print(train_dataset)
print(len(train_dataset))





word_counter = Counter()
for word in train_dataset['text'][:200000]:
    word =  word_tokenize(str(word).lower())
    word_counter.update(word)


print("vocab size before filtering ")
special_token  = { "<pad>":0,"<unk>": 1}
filtered = [(tok,count) for tok , count in word_counter.items() if count >=2]
#filtering the top k words cause it will take forever for it to encode and stuff
filtered =  sorted(filtered, key = lambda x : -x[1])[:top_k]

vocab  = {tok : i+len(special_token)  for i, (tok, _ )  in enumerate(filtered)}

print("Before merging ")
word2idx = {**special_token, **vocab}

#need reverse mapping for decoding
idx2word = {i: tok for tok , i in word2idx.items()}


def encode_text(text, max_length=None):
    words = word_tokenize(str(text).lower())
    words = words[:max_length]
    return [word2idx.get(word, word2idx["<unk>"]) for word in words]




def decode_text(indices):
    return " ".join(idx2word.get(idx, "<unk>") for idx in indices)



"""text = "The word is a great and happy place"

encoded = encode_text(text)
decoded = decode_text(encoded)

print(encoded)
print(decoded)"""
#need to create the batch to predict the next word
encoded_text = []
for i in train_dataset['text'][:200000]:
    encoded_text.extend(encode_text(i))

data  = torch.tensor(encoded_text, dtype=torch.long)

batch_size = 4#how many (x,y) pairs we
block_size = 16 # how many tokens will be there in the current window for context

#basically creating indexes to start the sequences
#like [1,4,7,10] means 4 sequences will start from these indexes
# generate random starting indexes

def get_batch(data, batch_size, block_size):
    ix  =  torch.randint(len(data) - block_size-1, (batch_size)) #pytorch expects a typle for the size
    # the first argument is the highest values (exclusive) and the second is the shape
    # so this many random numbers of
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #cause we did the out of bounds check in the ix thing we wont have that issue here





