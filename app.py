import os
import shutil
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------
# Startup Cleanup
# -----------------------

def startup_cleanup():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

startup_cleanup()

# -----------------------
# Load Environment
# -----------------------

load_dotenv()
groq_api_key = os.getenv("Groq_API_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

print("LLM Initialized Successfully")

# -----------------------
# Loaders
# -----------------------

def load_all_files():
    loader = DirectoryLoader(
        path="Books",
        glob="**/*",
        show_progress=True
    )
    return loader.load()

# -----------------------
# Chunking
# -----------------------

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# -----------------------
# Metadata Cleaning
# -----------------------

def clean_metadata(docs):
    for d in docs:
        d.metadata = {}
    return docs

# -----------------------
# Embeddings
# -----------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -----------------------
# Vector DB
# -----------------------

def create_vector_db(chunks, embeddings, collection_name):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    vectordb.persist()
    return vectordb

# -----------------------
# Build Pipeline
# -----------------------

documents = load_all_files()
documents = clean_metadata(documents)

chunks = split_docs(documents)

embeddings = get_embeddings()

collection_name = str(uuid.uuid4())

vectordb = create_vector_db(chunks, embeddings, collection_name)

print("Vector Database Built Successfully")

