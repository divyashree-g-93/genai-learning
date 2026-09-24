from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("document.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Split document into chunks
chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]

# Create embeddings for all document chunks
chunk_embeddings = model.encode(chunks)

# Ask a question
question = input("Ask a question: ")

# Create embedding for the question
question_embedding = model.encode([question])

# Calculate similarity
similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

# Sort chunks by similarity
results = sorted(
    zip(similarities, chunks),
    reverse=True
)

# Display top 3 results
print("\n--- Most Relevant Chunks ---")

for score, chunk in results[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print(chunk)