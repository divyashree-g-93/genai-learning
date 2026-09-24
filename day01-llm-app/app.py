import ollama

system_instruction = """
You are a senior AWS architect.

Your job is to provide practical, enterprise-focused AWS guidance.

When answering:
- Explain the architecture clearly.
- Mention relevant AWS services.
- Consider security, scalability, reliability, and cost.
- Use examples where useful.
- Keep the answer technically accurate and easy to understand.
"""

question = input("Ask your AWS question: ")

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": question
        }
    ],
    options={
        "temperature": 0.2
    }
)

print("\nAnswer:")
print(response["message"]["content"])