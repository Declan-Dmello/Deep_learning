from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class TranslationDataset(Dataset):
    def __init__(self, split="train", src_lang="en", tgt_lang="es", max_len=30):
        self.dataset = load_dataset("opus_books", f"{src_lang}-{tgt_lang}", split=split)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

        # Build vocabs
        self.src_vocab = self.build_vocab([ex["translation"][self.src_lang] for ex in self.dataset])
        self.tgt_vocab = self.build_vocab([ex["translation"][self.tgt_lang] for ex in self.dataset])

        # Reverse lookup (optional debugging)
        self.src_inv_vocab = {v: k for k, v in self.src_vocab.items()}
        self.tgt_inv_vocab = {v: k for k, v in self.tgt_vocab.items()}

    def build_vocab(self, texts):
        tokens = set()
        for t in texts:
            tokens.update(t.lower().split())
        vocab = {tok: idx + 4 for idx, tok in enumerate(sorted(tokens))}
        vocab["<pad>"] = 0
        vocab["<sos>"] = 1
        vocab["<eos>"] = 2
        vocab["<unk>"] = 3
        return vocab

    def encode(self, text, vocab):
        tokens = text.lower().split()[: self.max_len - 2]  # reserve for <sos>, <eos>
        return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx]["translation"][self.src_lang]
        tgt_text = self.dataset[idx]["translation"][self.tgt_lang]

        src = self.encode(src_text, self.src_vocab)
        tgt = self.encode(tgt_text, self.tgt_vocab)

        # target has <sos> at start and <eos> at end
        tgt = [self.tgt_vocab["<sos>"]] + tgt + [self.tgt_vocab["<eos>"]]

        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def get_dataloader(split="train", batch_size=32, src_lang="en", tgt_lang="es"):
    dataset = TranslationDataset(split, src_lang, tgt_lang)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), dataset
