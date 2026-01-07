# 📄 Insurance Policy RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot for **Insurance Policy Question Answering**.  
The chatbot answers user questions **strictly using uploaded insurance policy PDF documents**, without hallucination or external knowledge.

---

## 🚀 Features

- 📚 Load and process multiple insurance policy PDFs
- ✂️ Chunk documents with overlap for better retrieval
- 🧠 Create embeddings using **Ollama (`nomic-embed-text`)**
- 🗂 Store embeddings in **ChromaDB**
- 🤖 Answer queries using **Groq LLM (`openai/gpt-oss-20b`)**
- ❌ Explicitly refuses to answer if information is not present in documents
- 🧪 Includes test script for ingestion validation

---

## 📁 Project Structure

```

.
├── data/                    # Folder containing insurance PDF files
│   ├── policy1.pdf
│   └── policy2.pdf
│
├── chroma_db/               # Persisted Chroma vector database
│
├── ingest.py                # PDF loading + chunking + vector DB creation
├── chatbot.py               # Interactive RAG-based chatbot
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

````

---

## 🛠 Tech Stack

- **Python 3.10+**
- **LangChain**
- **ChromaDB**
- **Ollama**
- **Groq LLM**
- **PDFLoader**
- **RecursiveCharacterTextSplitter**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AsadCodeCraft/PDF_ChatBot_Using_RAG_LLM.git
cd PDF_ChatBot_Using_RAG_LLM
````

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install & Run Ollama

Download and install Ollama from:
👉 [https://ollama.com](https://ollama.com)

Pull the required embedding model:

```bash
ollama pull nomic-embed-text
```

Ensure Ollama is running:

```bash
ollama run nomic-embed-text
```

---

## 📥 Add Insurance Documents

Place all insurance policy PDFs inside the `data/` folder:

```
data/
├── health_policy.pdf
├── motor_policy.pdf
```

---

## 🧠 Create Vector Database

Run the ingestion script:

```bash
python ingest.py
```

✅ This will:

* Load all PDFs
* Split them into chunks
* Generate embeddings
* Persist them into `chroma_db/`

---

## 🤖 Run the Chatbot

```bash
python chatbot.py
```

Example interaction:

```
🤖 Insurance Policy Q&A Chatbot
Type 'exit' to quit

You: What is the waiting period for hospitalization?

Answer:
The waiting period for hospitalization is 30 days.

--------------------------------------------------
```

If information is **not found**, the chatbot responds **exactly** with:

```
The provided insurance documents do not contain this information.
```

---


## 📌 Design Principles

* ❌ No hallucination
* ❌ No assumptions
* ❌ No external insurance knowledge
* ✅ Answers strictly grounded in document context
* ✅ Deterministic output (`temperature=0`)

---

## 📈 Future Enhancements

* 🌐 Web UI (Streamlit / FastAPI)
* 📊 Source citations per answer
* 🔎 Metadata filtering
* 🧠 Multi-model support
* ☁️ Cloud deployment (Docker)

---

## 📜 License

MIT License
