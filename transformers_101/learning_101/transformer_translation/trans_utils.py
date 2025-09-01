import torch

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=40, device="cpu"):
    model.eval()
    idx2word = {i: w for w, i in tgt_vocab.items()}

    src = src.unsqueeze(0).to(device)
    tgt = torch.tensor([[tgt_vocab["<sos>"]]], device=device)

    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == tgt_vocab["<eos>"]:
            break

    words = [idx2word[idx.item()] for idx in tgt[0][1:]]
    return " ".join([w for w in words if w not in ["<eos>", "<pad>"]])
