import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import CTransformers
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, Index

# ================================
# Step 1: Load PDF Documents
# ================================
def load_pdf(data_path):
    """Load all PDF documents from the specified directory."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf("data/")

# ================================
# Step 2: Split Text into Chunks
# ================================
def text_split(extracted_data):
    """Split the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)
print("Length of the chunks: ", len(text_chunks))

# ================================
# Step 3: Load Hugging Face Embeddings
# ================================
def download_hugging_face_embeddings():
    """Download and initialize Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello World")
print("Length of embedding vector: ", len(query_result))  # Expected length: 384

# ================================
# Step 4: Configure Pinecone
# ================================

# Set API key for Pinecone
api_key = "pcsk_5ocNt6_6E4Gy6T3EUt8aSyv2fVBXLpTXyDAJXsfQjbMx3Tp2WKYkGuUPDpAr1NSQDNyQDM"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Index setup
index_name = "medical-chatbot"

# Check if index exists, create if it doesn't
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=len(query_result),
        metric="cosine"
    )

# Initialize the vector store with PineconeVectorStore
vector_store = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=api_key
)

# ================================
# Step 5: Define Query and Prompt Template
# ================================
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else...
Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ================================
# Step 6: Load LLM Model
# ================================
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8}
)

# ================================
# Step 7: Initialize QA Chain
# ================================
# Create a document chain that combines retrieved documents with the query
document_chain = create_stuff_documents_chain(llm, PROMPT)

# Create a retrieval chain that retrieves documents and then applies the document chain
retriever = vector_store.as_retriever(search_kwargs={'k': 2})
qa_chain = create_retrieval_chain(retriever, document_chain)

# ================================
# Step 8: Run Chatbot Loop
# ================================
while True:
    user_input = input("Input Prompt: ")
    result = qa_chain.invoke({"question": user_input})
    print("Response: ", result["answer"])