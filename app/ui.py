import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Define the Custom Embedder
class GeminiEmbeddings(Embeddings):
    def __init__(self, key):
        self.client = genai.Client(api_key=key)
    def embed_documents(self, texts):
        return [self.client.models.embed_content(model="gemini-embedding-2", contents=t).embeddings[0].values for t in texts]
    def embed_query(self, text):
        return self.client.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values

# 3. Cache the Database Connection
@st.cache_resource
def load_system():
    embedding_function = GeminiEmbeddings(key=api_key)
    database = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    ai_client = genai.Client(api_key=api_key)
    return database, ai_client

db, client = load_system()

# --- WEB INTERFACE DRAWING ---

st.title("🌊 VectorFlow HR Bot")
st.caption("Your strict, professional AI assistant for company policy.")

# Initialize the chat history in the browser's memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all past messages every time the page updates
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box at the bottom of the screen
if user_question := st.chat_input("Ask about office hours, equipment, or pets..."):
    
    # 1. Draw the user's message on the screen and save it
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # 2. Draw the AI's response on the screen
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            
            # Retrieve context
            docs = db.similarity_search(user_question, k=1)
            context = docs[0].page_content
            
            # Build strict prompt
            prompt = f"""You are a strict, professional HR assistant for VectorFlow Corporation. 
            You must answer the employee's question using ONLY the provided context below.
            If the context does not contain the answer, you must reply exactly with: "I'm sorry, but I can only answer questions related to the VectorFlow company policy."
            Do NOT use your general knowledge.
            
            Context: {context}
            Question: {user_question}"""
            
            # Call Gemini
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            # Show the result
            st.markdown(response.text)
            
    # 3. Save the AI's response to the memory
    st.session_state.messages.append({"role": "assistant", "content": response.text})