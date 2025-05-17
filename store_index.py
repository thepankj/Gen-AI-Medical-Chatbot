from src.helper import pdf_loader_from_directory, text_splitter, get_embedding_model
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot-db"
pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

data_path = r"Data/"
extracted_data = pdf_loader_from_directory(data_path)
text_chunks = text_splitter(extracted_data)
embedding = get_embedding_model()

index_name = "medibot-db"
doc_vector = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embedding, index_name=index_name)