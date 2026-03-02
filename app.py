import os
import shutil
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Startup Cleanup

def startup_cleanup():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

startup_cleanup()

# Load Environment

load_dotenv()
groq_api_key = os.getenv("Groq_API_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

print("LLM Initialized Successfully")

# Loaders

def load_all_files():
    loader = DirectoryLoader(
        path="Books",
        glob="**/*",
        show_progress=True
    )
    return loader.load()

# Chunking

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# Metadata Cleaning

def clean_metadata(docs):
    for d in docs:
        d.metadata = {}
    return docs

# Embeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Vector DB

def create_vector_db(chunks, embeddings, collection_name):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    vectordb.persist()
    return vectordb

# Build Pipeline

documents = load_all_files()
documents = clean_metadata(documents)

chunks = split_docs(documents)

embeddings = get_embeddings()

collection_name = str(uuid.uuid4())

vectordb = create_vector_db(chunks, embeddings, collection_name)

print("Vector Database Built Successfully")

# Retriever

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Retriever Initialized")

def retrieve_context(query):
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context, docs

def build_prompt(context, question):

    prompt = f"""
You are a research assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt

def generate_answer(question):

    context, docs = retrieve_context(question)

    prompt = build_prompt(context, question)

    response = model.invoke(prompt)

    return response.content, docs

while True:

    query = input("\nEnter your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer, sources = generate_answer(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources Used:\n")

    for i, doc in enumerate(sources):
        print(f"Source {i+1}")
        print(doc.page_content[:200])
        print("----------------------")
