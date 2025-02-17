from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

# Loading the data and performing chunks and finally doing the embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initializing Pinecone and creating embeddings for it
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name="medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)