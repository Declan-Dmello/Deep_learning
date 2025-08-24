from dataset_file import encode_text, word2idx
from model import TransformerClassifier
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#passing the vocab size i.e the unique word2idx list
model = TransformerClassifier(vocab_size=len(word2idx))

#loading the saved model
model.load_state_dict(torch.load("Customer_Sentiment_model.pt"))
model.to(device)


def predict_sentiment(model , text):
    model.eval()
    tokens = torch.tensor(encode_text(text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tokens)
        return "Positive" if pred.argmax(1).item() == 1 else "Negative"


print(predict_sentiment(model, "The Sky is beautiful"))


