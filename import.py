import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to initialize Selenium WebDriver
def initialize_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    return driver

# Function to scrape Google and retrieve relevant articles
def scrape_articles(search_query, output_folder="./docs", processed_folder="./processed_docs"):
    driver = initialize_driver()
    try:
        # Go to Google and perform search
        driver.get("https://www.google.com")
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)

        # Collect article links
        links = driver.find_elements(By.CSS_SELECTOR, "a")
        article_urls = [
            link.get_attribute("href") 
            for link in links 
            if link.get_attribute("href") and "http" in link.get_attribute("href")
            and is_valid_article_link(link.get_attribute("href"))
        ]

        # Create output folders if not exist
        for folder in [output_folder, processed_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Create topic-specific subfolder
        topic_folder = os.path.join(processed_folder, sanitize_filename(search_query))
        if not os.path.exists(topic_folder):
            os.makedirs(topic_folder)

        # Visit each article and scrape content
        for idx, url in enumerate(article_urls[:5]):
            try:
                driver.get(url)
                time.sleep(2)
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Extract main content
                paragraphs = soup.find_all("p")
                content = "\n".join([para.text for para in paragraphs])

                # Save raw content first
                raw_file_path = os.path.join(output_folder, f"article_{idx + 1}.txt")
                with open(raw_file_path, "w", encoding="utf-8") as file:
                    file.write(content)

                # Process and extract relevant information
                if check_relevance(search_query, content):
                    relevant_content = extract_relevant_information(search_query, content)
                    if relevant_content:
                        processed_file_path = os.path.join(
                            topic_folder, 
                            f"processed_article_{idx + 1}.txt"
                        )
                        with open(processed_file_path, "w", encoding="utf-8") as file:
                            file.write(relevant_content)
                        print(f"Processed and saved: {processed_file_path}")

            except Exception as e:
                print(f"Error processing URL {url}: {e}")
    finally:
        driver.quit()

def sanitize_filename(filename):
    """Convert search query to valid folder name"""
    return "".join(c if c.isalnum() else "_" for c in filename).lower()

def is_valid_article_link(url):
    exclusion_patterns = [
        "help", "support", "contact", "accessibility", "google.com", 
        "forum", "advertisement", "login"
    ]
    if any(pattern in url.lower() for pattern in exclusion_patterns):
        return False
    if "article" in url or "blog" in url:
        return True
    return False

def check_relevance(query, content):
    llm = OllamaLLM(model="llama3.1:latest")
    prompt = f"""
    Analyze if the following content is relevant to the topic: '{query}'
    Consider:
    1. Topic alignment
    2. Information value
    3. Reliability of information

    Content:
    {content[:1000]}...

    Respond with Yes or No and a brief explanation.
    """
    response = llm.invoke(prompt)
    return "yes" in response.lower()

def extract_relevant_information(query, content):
    llm = OllamaLLM(model="llama3.1:latest")
    
    # Split content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(content)
    
    relevant_chunks = []
    for chunk in chunks:
        prompt = f"""
        Extract and summarize only the information that is directly relevant to: '{query}'
        
        Text to analyze:
        {chunk}
        
        Instructions:
        1. Focus only on information directly related to the topic
        2. Maintain technical accuracy
        3. Remove any irrelevant examples or tangents
        4. Preserve key concepts and terminology
        5. Format in clear, concise paragraphs
        
        Relevant information:
        """
        
        response = llm.invoke(prompt)
        if response.strip():
            relevant_chunks.append(response)
    
    # Combine and structure the relevant information
    if relevant_chunks:
        final_prompt = f"""
        Organize and structure the following information about '{query}':

        {' '.join(relevant_chunks)}

        Please:
        1. Remove any redundant information
        2. Ensure logical flow
        3. Add section headings where appropriate
        4. Maintain technical accuracy
        """
        
        structured_content = llm.invoke(final_prompt)
        return structured_content
    
    return None

def main():
    queries = [
        "Agile project management basics",
        "Software development lifecycle best practices",
        "How to create a project quotation",
        "FAQs for IT clients"
    ]
    for query in queries:
        print(f"\nProcessing query: {query}")
        scrape_articles(query)

if __name__ == "__main__":
    main()