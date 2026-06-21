# ============================================
# chatbot.py - Core chatbot logic
# ============================================

# Step 1: Import libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Step 2: Load API key from .env file
load_dotenv()

# -----------------------------------------------
# Step 3: Load the vector database
# -----------------------------------------------
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # Load the saved FAISS database from disk
    vectordb = FAISS.load_local(
        "vectordb",
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading saved FAISS
    )
    return vectordb

# -----------------------------------------------
# Step 4: Create the chatbot chain
# -----------------------------------------------
def create_chatbot():

    # Load vector database
    vectordb = load_vectordb()

    # Create retriever - searches vectordb for relevant chunks
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )

    # Load Gemini LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2  # Low temperature = more focused, less creative answers
                         # High temperature = more random answers (bad for support bot)
    )

    # -----------------------------------------------
    # Step 5: Add Memory (Chat History)
    # -----------------------------------------------
    # Memory saves previous questions and answers
    # So chatbot remembers what was said before

    memory = ConversationBufferMemory(
        memory_key="chat_history",   # Key name for storing history
        return_messages=True,        # Return full message objects
        output_key="answer"          # Save only the answer in memory
    )

    # -----------------------------------------------
    # Step 6: Create the full RAG chain with hallucination control
    # -----------------------------------------------
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Show which document was used
        verbose=False
    )

    return chain

# -----------------------------------------------
# Step 7: Hallucination Control Prompt
# -----------------------------------------------
# This is the most important part of Project 2!
# We tell the LLM to ONLY answer from the given context

SYSTEM_PROMPT = """
You are a helpful customer support assistant.
IMPORTANT RULES:
1. Only answer questions based on the provided context/documents.
2. If the answer is not in the context, say exactly: 
   "I'm sorry, I don't have information about that in my knowledge base."
3. Never make up or guess an answer.
4. Keep answers short and clear.
"""

def get_answer(chain, question):
    # Combine system prompt with user question
    full_question = SYSTEM_PROMPT + "\n\nUser Question: " + question

    # Get answer from chain
    result = chain({"question": full_question})

    return result["answer"]
