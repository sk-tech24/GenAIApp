import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from threading import Thread
import os
from utils import config

# Initialize the embedding model
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to extract all URLs from a website
def extract_dynamic_urls(url, collected_urls):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector('a')  # Wait for anchor tags to load
        links = page.query_selector_all('a[href]')
        urls = [urljoin(url, link.get_attribute('href')) for link in links]
        unique_urls = set(urls)
        collected_urls.update(unique_urls)
        browser.close()

# Function to scrape website data and store it in a list
def scrape_and_store_data(start_url):
    urls_to_scrape = set([start_url])
    collected_urls = set()
    threads = []

    while urls_to_scrape:
        url = urls_to_scrape.pop()
        thread = Thread(target=extract_dynamic_urls, args=(url, collected_urls))
        threads.append(thread)
        thread.start()
        if len(threads) > 5:
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.join()

    print(f"Collected {len(collected_urls)} unique URLs")
    
    clean_urls = [u for u in collected_urls if u.startswith("https")]
    loader = WebBaseLoader(clean_urls)
    docs = loader.load()
    
    return docs

# Function to embed documents using Hugging Face and store embeddings in FAISS
def embed_and_store(docs):
    # Create a list to store embeddings and metadata
    all_embeddings = []
    all_metadata = []

    # Batch processing for embedding documents
    batch_size = 100  # Adjust batch size based on your memory
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        texts = [doc.page_content for doc in batch_docs]
        
        # Get embeddings for the batch of documents
        embeddings_batch = embedding_model.encode(texts, convert_to_numpy=True)
        all_embeddings.extend(embeddings_batch)
        
        # Store metadata for each document (e.g., URL, content)
        all_metadata.extend([doc.metadata for doc in batch_docs])
    
    # Convert embeddings to a numpy array for FAISS
    embeddings_np = np.array(all_embeddings).astype(np.float32)

    # Initialize FAISS index
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)  # Use L2 distance for similarity search

    # Add embeddings to the FAISS index
    index.add(embeddings_np)
    
    print(f"Stored {len(embeddings_np)} vectors in FAISS")

    return index, all_metadata

# Function to perform retrieval from FAISS
def search_faiss(query, index, metadata, k=5):
    # Get embedding for the query
    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    
    # Perform search in FAISS
    distances, indices = index.search(query_vector, k)
    
    # Retrieve the metadata of the most similar documents
    results = []
    for i in indices[0]:
        results.append(metadata[i])
    
    return results

# Main script to scrape, embed, and store in FAISS
if __name__ == "__main__":
    # Step 1: Scrape the website and get documents
    start_url = config.URL
    print("Starting to scrape the website and load documents...")
    docs = scrape_and_store_data(start_url)
    
    # Step 2: Embed the documents and store them in FAISS
    print("Embedding documents and storing them in FAISS...")
    index, metadata = embed_and_store(docs)
    
    # Step 3: Initialize the RetrievalQA chain
    retriever = FAISS(index).as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=None,  # Replace with your chosen LLM
        chain_type="stuff",
        retriever=retriever,
    )
    
    # Step 4: Start the interactive loop for user queries
    print("Ready to accept user queries. Type 'quit' to exit.")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "quit":
            print("Exiting the application. Goodbye!")
            break
        
        # Process the query through the RAG pipeline
        result = qa_chain.run(query)
        
        # Print the answer to the user's query
        print("Answer:", result)
