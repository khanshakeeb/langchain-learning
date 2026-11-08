from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Local embedding model served by Ollama (run `ollama pull nomic-embed-text` once)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

COLLECTION_NAME = "rag-chroma"
PERSIST_DIRECTORY = "./.chroma"
BATCH_SIZE = 32

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def ingest() -> None:
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )._collection.count()
    if existing:
        print(
            f"{PERSIST_DIRECTORY} already has {existing} chunks, skipping. "
            f"Delete {PERSIST_DIRECTORY} to re-index."
        )
        return

    docs =[WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    # Embed in small batches: sending all chunks to Ollama in one request
    # can intermittently fail with "connection reset by peer".
    for i in range(0, len(doc_splits), BATCH_SIZE):
        vectorstore.add_documents(doc_splits[i : i + BATCH_SIZE])
    print(f"Indexed {len(doc_splits)} chunks into {PERSIST_DIRECTORY}")


retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
).as_retriever()


if __name__ == "__main__":
    ingest()
