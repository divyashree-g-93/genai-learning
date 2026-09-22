import ollama

question = input("Ask a question: ")

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\nAnswer:")
print(response["message"]["content"])