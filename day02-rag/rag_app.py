from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("document.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Split into chunks
chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]

# Create embeddings for document chunks
chunk_embeddings = embedding_model.encode(chunks)

# Ask question
question = input("Ask a question: ")

# Create question embedding
question_embedding = embedding_model.encode([question])

# Calculate similarity
similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

# Get top 3 chunks
results = sorted(
    zip(similarities, chunks),
    reverse=True
)

print("\n--- Retrieved Context ---")

for score, chunk in results[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print(chunk)

relevant_chunks = [
    chunk for score, chunk in results[:5]
]

# Combine retrieved chunks
context = "\n\n".join(relevant_chunks)

# Create RAG prompt
prompt = f"""
Answer the question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}
"""

# Send to Llama
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0.2
    }
)

print("\n--- Answer ---")
print(response["message"]["content"])