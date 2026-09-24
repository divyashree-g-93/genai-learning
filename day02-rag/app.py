import ollama


def load_document(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text):
    return [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]


document = load_document("document.txt")
chunks = create_chunks(document)

print(chunks)
print(f"Loaded {len(chunks)} document chunks.")

question = input("\nAsk a question about the document: ")

# Simple keyword-based retrieval
question_words = set(question.lower().split())

stop_words = {
    "the", "is", "a", "an", "how", "what", "why",
    "can", "i", "do", "to", "for", "of", "and",
    "my", "in", "on", "should"
}

question_words = question_words - stop_words

scored_chunks = []

for chunk in chunks:
    chunk_words = set(chunk.lower().split())

    score = len(question_words.intersection(chunk_words))

    if score > 0:
        scored_chunks.append((score, chunk))

# Sort chunks by relevance score
scored_chunks.sort(reverse=True)

# Take only the top 3
relevant_chunks = [
    chunk for score, chunk in scored_chunks[:3]
]

context = "\n\n".join(relevant_chunks)

prompt = f"""
Answer the user's question using ONLY the context provided below.

If the answer cannot be found in the context, say:
"I don't have enough information in the document."

Context:
{context}

Question:
{question}
"""
print("\n--- Retrieved Context ---")
print(context)
print("-------------------------")
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

print("\nAnswer:")
print(response["message"]["content"])