import os
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from environment variables
aws_access_key = os.getenv("ACCESS_KEY")
aws_secret_key = os.getenv("SECRET_KEY")

# Set AWS credentials as environment variables
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

def create_pdf_qa_system(pdf_path):
    # Step 1: Load the PDF
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the PDF.")
    
    # Step 2: Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Step 3: Initialize Bedrock embeddings
    print("Initializing Bedrock embeddings...")
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )
    
    # Step 4: Create a FAISS vector store from the chunks
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")
    
    print("PDF QA system components created successfully!")
    return vector_store

def ask_question(vector_store, llm, question, k=3):
    print(f"Question: {question}")
    
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(question, k=k)
    
    # Format the context from retrieved documents
    context = "\n\n".join([f"Document (Page {doc.metadata['page']}):\n{doc.page_content}" for doc in retrieved_docs])
    
    # Create prompt with context and question
    prompt = f"""I need you to answer a question based on the following document excerpts.

        Context:
        {context}

        Question: {question}

        Please provide a comprehensive answer based only on the information in these document excerpts. If the answer is not contained in the provided context, please state that you don't have enough information to answer accurately.
    """
    
    # Invoke the model with the prompt
    response = llm.invoke(prompt)
    
    print(f"Answer: {response.content}")
    
    return {
        "result": response.content,
        "source_documents": retrieved_docs
    }

# Example usage
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_path = "resume.pdf"
    
    # Create the QA system
    vector_store = create_pdf_qa_system(pdf_path)
    
    # Initialize the Bedrock LLM using langchain_aws.ChatBedrock
    print("Initializing Bedrock LLM...")
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1"
    )

    # Ask questions
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        ask_question(vector_store, llm, question)