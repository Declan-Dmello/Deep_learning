from sentence_transformers import SentenceTransformer
import faiss
import pickle

documents = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "Python is a popular programming language for AI.",
    "FAISS is a library for efficient similarity search."
]

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, convert_to_numpy=True)

# Build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Save index + docs
faiss.write_index(index, "faiss_index.idx")
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Index built and saved!")
