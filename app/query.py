import os
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

# We reuse the same custom Embeddings class from our ingestion script
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key = api_key)
    def embed_documents(self, texts):
        return [self.client.models.embed_content(model = "gemini-embedding-2", contents = t).embeddings[0].values for t in texts]
    def embed_query(self, text):
        return self.client.models.embed_content(model = "gemini-embedding-2", contents = text).embeddings[0].values

# 1. Connect to the existing Database
print("Loading database...")
embedding_function = GeminiEmbeddings(api_key = os.getenv("GOOGLE_API_KEY"))
db = Chroma(persist_directory = "chroma_db", embedding_function = embedding_function)

# 2. Define the question
query = "Can I bring my iguana to the office?"
print(f"Searching for {query}")

# 3. Retrieve relevant chunks
docs = db.similarity_search(query, k = 1)
context = docs[0].page_content

# 4. Generate answer using Gemini
client = genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))
prompt = f"Use this context to answer the question. Context: {context}. Question: {query}"

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = prompt
)

print("\n--AI Answer---")
print(response.text)
