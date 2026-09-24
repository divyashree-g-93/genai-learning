from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "How can I allow another AWS account to access my S3 bucket?"

text2 = "For cross-account access, IAM roles and bucket policies can be used to grant controlled access to another AWS account."

text3 = "S3 Versioning can be enabled to preserve and recover previous versions of objects."

embedding1 = model.encode(text1)
embedding2 = model.encode(text2)
embedding3 = model.encode(text3)

print("Embedding size:", len(embedding1))