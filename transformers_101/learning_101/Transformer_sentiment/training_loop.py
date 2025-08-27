import torch
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from model import TransformerClassifier
from dataset_file import train_loader, encode_text, word2idx
print("Training loop")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerClassifier(vocab_size=len(word2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion= nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, )

print("Reached here")
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch : {epoch}, Loss : {avg_loss:.4f}, Learning_Rate ;{scheduler.get_last_lr()[0]}")
torch.save(model.state_dict(), "Customer_Sentiment_model.pt")

def predict_sentiment(model , text):

    model.eval()
    tokens = torch.tensor(encode_text(text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tokens)
        return "Positive" if pred.argmax(1).item() == 1 else "Negative"


print(predict_sentiment(model, "The world is a happy place"))



