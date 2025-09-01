import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from data import get_dataloader
from trans_utils import greedy_decode

def train_model(epochs=5, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):

    train_loader, dataset = get_dataloader("train", batch_size=batch_size)
    model = TransformerModel(len(dataset.src_vocab), len(dataset.tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # quick test translation
        test_src, _ = next(iter(train_loader))
        sentence = greedy_decode(model, test_src[0], dataset.src_vocab, dataset.tgt_vocab, device=device)
        print("Sample Translation:", sentence)

if __name__ == "__main__":
    train_model()
