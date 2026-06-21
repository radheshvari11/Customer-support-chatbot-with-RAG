# ============================================
# ingest.py - Load documents and create embeddings
# ============================================

# Step 1: Import all required libraries
from langchain_community.document_loaders import TextLoader  # Loads .txt files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # Creates embeddings
from langchain_community.vectorstores import FAISS  # Vector database
import os

# -----------------------------------------------
# Step 2: Load the document
# -----------------------------------------------
print("Loading document...")

loader = TextLoader("data/company_faq.txt")  # Load the FAQ text file
documents = loader.load()  # Read all content from file

print(f"Loaded {len(documents)} document(s)")

# -----------------------------------------------
# Step 3: Split text into small chunks
# -----------------------------------------------
# Why split? Because LLM cannot read entire document at once
# We break it into small pieces (chunks) for better search

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Each chunk = 200 characters
    chunk_overlap=50     # 50 characters overlap between chunks
                         # Overlap helps so meaning is not lost at boundaries
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# -----------------------------------------------
# Step 4: Create embeddings
# -----------------------------------------------
# Embedding = converts text into numbers (vectors)
# Similar meaning = similar numbers
# This helps find relevant answers to questions

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Small, fast, free embedding model
)

# -----------------------------------------------
# Step 5: Store embeddings in FAISS vector database
# -----------------------------------------------
print("Saving to vector database...")

vectordb = FAISS.from_documents(chunks, embeddings)
vectordb.save_local("vectordb")  # Save to 'vectordb' folder

print("Done! Vector database created successfully.")
