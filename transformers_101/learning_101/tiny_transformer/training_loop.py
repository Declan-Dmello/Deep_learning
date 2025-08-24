import torch
from dataset_encoding import encode, decode, vocab_size, data, CharDataset
from model import TinyTransformer
from torch.utils.data import DataLoader
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 8
batch_size = 8

dataset = CharDataset(data, block_size)
#this will build training examples

#Using Dataloader for batching of data
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = TinyTransformer(vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

criterion = nn.CrossEntropyLoss()



for epoch in range(200):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.step()
    if epoch% 10 ==0:
        print(f"Epoch {epoch}, loss {loss.item():4f}")



def generate(model, start_text ="to be", length= 10):

    model.eval()
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(length):
        logits = model(context [:, -block_size:])
        next_char_logits = logits[:,-1,:]
        #basically turns logits into probability,
        # -1 is basically like apply softmax to the last dimension of the tensor , we could also use 1 ,as it is doing it row wise
        #1 is basically the number of samples to return , more number would return more smaples
        next_char = torch.multinomial(torch.softmax(next_char_logits/0.7, dim=-1), 1)
        #multinomial is like weighted randomness, higher weight == higher change of coming
        context = torch.cat([context, next_char], dim=1)
    return decode(context[0].tolist())

print(generate(model))