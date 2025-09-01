import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# Load index & docs
index = faiss.read_index("faiss_index.idx")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# ==== CHANGE START: use FLAN-T5 for text2text ====
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=device
)
# ==== CHANGE END ====

def retrieve(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]

def rag_response(query):
    retrieved = retrieve(query)
    context = "\n".join(f"- {d}" for d in retrieved)
    prompt = (
        "Answer the question using only the context. "
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    out = generator(
        prompt,
        max_new_tokens=64,     # T5 uses max_new_tokens
        do_sample=False        # deterministic, reduces nonsense
    )
    return out[0]["generated_text"].strip()
