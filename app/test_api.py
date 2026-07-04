import os
from dotenv import load_dotenv
from google import genai

# 1. Load the secret key from your .env file
load_dotenv()
my_key = os.getenv("GOOGLE_API_KEY")

# 2. Initialize the Gemini client using the modern google-genai SDK
client = genai.Client(api_key=my_key)

# 3. Send a test message to the AI model
print("Connecting to Gemini...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello! This is VectorFlow speaking. Are you receiving me?"
)

# 4. Print the AI's reply to the terminal
print("\n--- AI Response ---")
print(response.text)
print("-------------------\n")