from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os

DATA_PATH = "data"
DB_PATH = "chroma_db"


def load_documents():
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    print(f"📂 Found {len(pdf_files)} PDF files")

    for idx, file in enumerate(pdf_files, start=1):
        print(f"📄 [{idx}/{len(pdf_files)}] Loading: {file}")
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs = loader.load()
        documents.extend(docs)
        print(f"\t|--> Loaded {len(docs)} pages")

    print(f"✅ Total pages loaded: {len(documents)}\n")
    return documents


def split_documents(documents):
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks\n")
    return chunks


def create_vector_db(chunks):
    print("🧠 Initializing embeddings model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("📦 Creating Chroma vector database with progress bar...")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    # 🔥 Chunk-level progress bar
    for chunk in tqdm(chunks, desc="🔗 Embedding chunks", unit="chunk"):
        vectordb.add_documents([chunk])

    vectordb.persist()
    print("\n💾 Vector database persisted successfully\n")


if __name__ == "__main__":
    print("🚀 Starting Vector DB creation process\n")

    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_db(chunks)

    print("🎉 Vector DB created successfully!")