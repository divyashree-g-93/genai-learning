from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# -----------------------------
# 1. Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Load document
# -----------------------------
with open("document.txt", "r", encoding="utf-8") as file:
    document = file.read()


# -----------------------------
# 3. Create document chunks
# -----------------------------
chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]


# -----------------------------
# 4. Create embeddings
# -----------------------------
print("Creating document embeddings...")

chunk_embeddings = embedding_model.encode(chunks)

print(f"Created embeddings for {len(chunks)} chunks.")


# -----------------------------
# 5. Interactive question loop
# -----------------------------
while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # -----------------------------
    # 6. Embed the question
    # -----------------------------
    question_embedding = embedding_model.encode([question])

    # -----------------------------
    # 7. Calculate similarity
    # -----------------------------
    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    # -----------------------------
    # 8. Sort results
    # -----------------------------
    results = sorted(
        zip(similarities, chunks),
        reverse=True
    )

    # -----------------------------
    # 9. Retrieve relevant chunks
    # -----------------------------
    SIMILARITY_THRESHOLD = 0.55

    relevant_results = [
    (score, chunk)
    for score, chunk in results[:5]
    if score >= SIMILARITY_THRESHOLD
]

    relevant_chunks = [
    chunk
    for score, chunk in relevant_results
    ]
    print("\n--- Retrieved Sources ---")

    for score, chunk in relevant_results:
        print(f"\nSimilarity: {score:.4f}")
        print(chunk)

    # -----------------------------
    # 10. Handle no relevant context
    # -----------------------------
    if not relevant_chunks:
        print("\nAnswer:")
        print("I don't have enough information in the provided document.")
        continue

    # -----------------------------
    # 11. Build context
    # -----------------------------
    context = "\n\n".join(relevant_chunks)

    # -----------------------------
    # 12. Build RAG prompt
    # -----------------------------
    prompt = f"""
You are an AWS assistant.

Answer the question using ONLY the information provided
in the context below.

Do not use your general knowledge.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{question}
"""

    # -----------------------------
    # 13. Call Llama
    # -----------------------------
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

    # -----------------------------
    # 14. Display answer
    # -----------------------------
    print("\nAnswer:")
    print(response["message"]["content"])