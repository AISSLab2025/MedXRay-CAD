from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

oembed = OllamaEmbeddings(model="mxbai-embed-large")

def mimic_db_retriever(query):
 
    vectordb = Chroma(
        persist_directory = "../data/embeddings/all_embed_db",
        embedding_function = oembed
    )
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    embeddings_filter = EmbeddingsFilter(embeddings=oembed, similarity_threshold=0.10)
    compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=retriever
            )
 
    compressed_docs = compression_retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in compressed_docs])
    
    return context