import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google import genai
from langchain_core.embeddings import Embeddings

load_dotenv()

# Custom Embeddings class to use the new 'google-genai' SDK directly
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    def embed_documents(self, texts):
        return [self.client.models.embed_content(model="gemini-embedding-2", contents=t).embeddings[0].values for t in texts]
    def embed_query(self, text):
        return self.client.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values

# 1. Load Data
loader = TextLoader("data/company_policy.txt")
documents = loader.load()

# 2. Split Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Create Database
print("Storing documents in Vector Database...")
embedding_function = GeminiEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"))
db = Chroma.from_documents(docs, embedding_function, persist_directory="chroma_db")
print("Done! Documents are stored.")