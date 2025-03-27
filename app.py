from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chains import create_retrieval_chain
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Function to download embeddings from helper.py
embeddings = download_hugging_face_embeddings()

index_name = "healthbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever for document search
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Custom GroqAI class to interact with Groq's API
class GroqAI:
    def __init__(self, api_key, temperature=0.4, max_tokens=500):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"  # Use Groq's actual endpoint

    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "llama-3.3-70b-versatile",  # Adjust this model as needed
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "n": 1  # Generating one completion
        }

        response = requests.post(self.api_url, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Error from Groq API: {response.text}")


# Initialize the Groq model with the API key
groq_llm = GroqAI(api_key=GROQ_API_KEY, temperature=0.4, max_tokens=500)

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the document chain and retrieval chain
question_answer_chain = create_stuff_documents_chain(groq_llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
