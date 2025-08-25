import torch
from sympy.core.random import shuffle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
#import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

#nltk.download('punkt')  # Download the punkt tokenizer if not already done
#print("Past the nltk download step")
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

#creating vocab from training text
word_counts = Counter()
for text in train_data["text"]:
    words = word_tokenize(str(text).lower())
    word_counts.update(words)

min_freq =2 #basically prevents single word from getting saved in the idx dict
#since its rare words prevents noise in the vocab and keeps it small
word2idx = {'<pad>': 0, '<unk>': 1}
#building word to index mapping
for word,count in word_counts.items():
    if count>= 2:
        word2idx[word] = len(word2idx)
        #basically adds the word to the dict if its not there already
        #basically gives a unique index to each word based on the length of the dict


def encode_text(text, max_length=None):
    words = word_tokenize(str(text).lower())
    words = words[:max_length]  # Truncate to max_length
    return [word2idx.get(word, word2idx['<unk>']) for word in words]
#basically the top word_counts and word2idx are used to create a dict
#and then using the mapping of the dict the encode text function is used
#basically creates the token ids for each word and also handles unknown words


#iterates thru the batch and then encodes it
#basically converts all text to indices and then tensor and basically gets the labels

def collate_batch(batch, max_length=None):
    labels, texts = [],[]
    for item in batch:
        text = item["text"]
        label = item["label"]
        labels.append(label)
        token_ids = torch.tensor(encode_text(text, max_length), dtype=torch.long)
        texts.append(token_ids)

    labels = torch.tensor(labels,dtype=torch.long)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    #will pad the sentences with 0 if they are shorter than the specified length
    return texts, labels

#reloads the train data cause we already used it for the vocab

#train_iter = dataset["train"]
train_loader = DataLoader(train_data,
                          collate_fn=lambda batch: collate_batch(batch, 512)
                          , shuffle=True, batch_size=32)

