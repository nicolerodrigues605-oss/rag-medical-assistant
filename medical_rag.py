import os
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------
# Step 1: Load Documents
# -------------------------------
def load_documents(folder_path="data"):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                documents.append(text)

    print("✅ Documents loaded")
    return documents

# -------------------------------
# Step 2: Split Documents
# -------------------------------
def split_documents(documents):
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# -------------------------------
# Step 3: Embeddings
# -------------------------------
def get_embeddings():
    print("🔄 Loading embeddings...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------------------
# Step 4: Create or Load FAISS
# -------------------------------
def get_vectorstore(embeddings):
    if os.path.exists("faiss_index"):
        print("✅ Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚡ Creating FAISS index (first time)...")

        docs = load_documents()
        chunks = split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")

    return vectorstore

# -------------------------------
# Step 5: LOAD FULL SYSTEM
# -------------------------------
def load_rag_system():
    print("🚀 Initializing RAG system...")

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("🔄 Loading LLM...")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    print("✅ RAG system ready!")
    return retriever, llm

# -------------------------------
# Step 6: MAIN FUNCTION
# -------------------------------
def ask_medical_question(query, retriever, llm):

    # Step 1: Retrieve docs
    docs = retriever.invoke(query)

    # Step 2: Create context
    context = "\n".join([doc.page_content for doc in docs])

    # Step 3: RAG prompt
    prompt = f"""
You are a helpful medical assistant.

Answer ONLY from the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    # 🔥 Step 4: FALLBACK TRIGGER
    if "I don't know" in answer or len(context.strip()) < 50:
        fallback_prompt = f"""
You are a helpful medical assistant.

Answer the question clearly:

Question: {query}
"""
        fallback_response = llm.invoke(fallback_prompt)
        return fallback_response.content

    return answer
# -------------------------------
# OPTIONAL: Terminal testing ONLY
# -------------------------------
if __name__ == "__main__":
    retriever, llm = load_rag_system()

    while True:
        query = input("Ask a medical question (type 'exit' to stop): ")
        if query.lower() == "exit":
            break

        answer = ask_medical_question(query, retriever, llm)
        print("\nAI Answer:\n", answer)