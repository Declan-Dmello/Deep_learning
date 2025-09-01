import torch
from model import TransformerModel
from data import pairs, collate_fn, SRC_VOCAB, TGT_VOCAB, TGT_IVOCAB
from trans_train import train

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare data
batch = collate_fn(pairs)

# Model
model = TransformerModel(len(SRC_VOCAB), len(TGT_VOCAB)).to(device)

# Training
train(model, [batch], epochs=10, device=device)

# Inference (simple greedy decoding)
def translate(sentence):
    src, _ = collate_fn([(sentence, "")])
    src = src.to(device)

    tgt = torch.tensor([[TGT_VOCAB["<sos>"]]], device=device)
    for _ in range(5):
        logits = model(src, tgt)
        next_word = logits[:, -1].argmax(dim=-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == TGT_VOCAB["<eos>"]:
            break

    return " ".join([TGT_IVOCAB[i] for i in tgt[0].tolist() if i > 2])

print("Translation:", translate("hello world"))
