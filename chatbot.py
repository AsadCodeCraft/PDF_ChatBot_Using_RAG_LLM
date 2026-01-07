# imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os
os.environ["GROQ_API_KEY"] = "your_api_key"

DB_PATH = "chroma_db"

# prompt (aligned with your notebook)
insurance_prompt = PromptTemplate.from_template("""
You are an insurance policy assistant. Answer the user’s question strictly and only using the information provided in the supplied policy documents.
Do not make assumptions. Do not use external knowledge or general insurance expertise.
Do not infer or extrapolate beyond what is explicitly stated in the documents.

If the requested information is not explicitly found in the provided policy context, respond exactly with:
“The provided insurance documents do not contain this information.”
                                

Context:
{context}

Question:
{question}

Answer:
""")

# RAG loader
def load_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print(retriever)

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
    )

    # Runnable RAG chain (same pattern as notebook)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | insurance_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# MAIN ENTRY POINT
def main():
    rag_chain, retriever = load_rag_chain()

    print("🤖 Insurance Policy Q&A Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer = rag_chain.invoke(query)

        print("\nAnswer:")
        print(answer)

        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
