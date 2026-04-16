# 🩺 RAG Medical Assistant

A Retrieval-Augmented Generation (RAG) based medical assistant that uses Groq LLM and a vector database to provide accurate, context-aware answers to medical queries.

---

## 🚀 Features

* 🔍 Context-aware medical question answering
* 📄 Uses custom medical dataset (text files)
* 🤖 Powered by Groq LLM
* ⚡ Fast semantic search using vector embeddings
* 💬 Simple interactive UI (Streamlit)

---

## 🛠️ Tech Stack

* Python
* Streamlit (for UI)
* Groq LLM
* RAG (Retrieval-Augmented Generation)
* Embeddings + Vector Database (for semantic search)

---

## 📂 Project Structure

```
rag-medical-assistant/
│
├── app.py                # Streamlit UI
├── medical_rag.py        # RAG pipeline logic
├── data/                 # Medical dataset
│   ├── Anemia.txt
│   ├── Blood_pressure.txt
│   └── ...
├── README.md
└── LICENSE
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/rag-medical-assistant.git
cd rag-medical-assistant
```

---

### 2. Install dependencies

If you have requirements.txt:

```
pip install -r requirements.txt
```

If not, install manually:

```
pip install streamlit langchain groq sentence-transformers faiss-cpu python-dotenv
```

---

### 3. Add API Key

Create a `.env` file in the root folder and add:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4. Run the application

Since this is a Streamlit app, run:

```
streamlit run app.py
```

---

## 🧠 How It Works (RAG Flow)

1. User enters a medical query
2. Query is converted into embeddings
3. Relevant documents are retrieved from dataset
4. Context + query is sent to Groq LLM
5. LLM generates a context-aware response

---

## 📌 Use Case

This project demonstrates how RAG can be applied in healthcare to build intelligent assistants that provide accurate and context-based information.

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical advice.

---

## 📬 Author

Nicole Rodrigues

