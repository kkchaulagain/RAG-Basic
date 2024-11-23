from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
model_name = "llama3.2:latest"
OLLAMA_SERVER_URL = "http://192.168.1.69:11434"
def main():
    # Step 1: Load documents
    docs_directory = "./docs"  # Update with the path to your project documents
    loader = DirectoryLoader(docs_directory)
    documents = loader.load()

    # Step 2: Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # Step 3: Set up the Conversational QA chain
    retriever = vector_store.as_retriever()
    # llm = Ollama(model=model_name,)
    llm = OllamaLLM(model=model_name, base_url=OLLAMA_SERVER_URL)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Step 4: Run chatbot
    print("Chatbot is live. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        answer = qa_chain.run(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()
