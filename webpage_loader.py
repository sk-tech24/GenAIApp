import os
import faiss
import numpy as np
from langchain.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.llms import Cohere
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from threading import Thread
from utils import config  # Replace with your actual configuration module

os.environ["COHERE_API_KEY"] = config.COHERE_API_KEY

# Function to extract all URLs from a website
def extract_dynamic_urls(url, collected_urls):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector('a')  # Wait for anchor tags to load
        links = page.query_selector_all('a[href]')
        urls = [urljoin(url, link.get_attribute('href')) for link in links]
        collected_urls.update(set(urls))
        browser.close()

# Function to scrape website data and store it in a list
def scrape_and_store_data(start_url):
    urls_to_scrape = {start_url}
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

# Function to embed documents using Cohere and store embeddings in FAISS
def embed_and_store(docs):
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,  # Replace with your Cohere model name
        user_agent="langchain"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Stored {len(docs)} documents in FAISS")

    return vectorstore

# Main script to scrape, embed, and set up RAG pipeline
if __name__ == "__main__":
    # Step 1: Scrape the website and get documents
    print("Starting to scrape the website and load documents...")
    docs = scrape_and_store_data(config.URL)

    # Step 2: Embed the documents and store them in FAISS
    print("Embedding documents and storing them in FAISS...")
    vectorstore = embed_and_store(docs)

    # Step 3: Initialize the language model (Cohere LLM)
    llm = Cohere(model=config.CHAT_MODEL)  # Use Cohere's LLM

    # Step 4: Define the RAG pipeline with LangChain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Step 5: Start the interactive loop for user queries
    print("Ready to accept user queries. Type 'quit' to exit.")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "quit":
            print("Exiting the application. Goodbye!")
            break

        # Process the query through the RAG pipeline
        result = qa_chain.run(query)

        # Print the answer to the user's query
        print("Answer:", result["answer"])
        print("Source Documents:", result.get("source_documents", []))
