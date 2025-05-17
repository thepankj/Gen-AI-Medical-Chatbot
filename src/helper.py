from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def pdf_loader_from_directory(data):
    """
    Loading all the PDF files from the directory
    """
    loader = DirectoryLoader(path=data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_splitter(extracted_data):
    """
    Splitting the loaded documents and creating chunks
    """
    text_splitter_obj = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    splitted_data = text_splitter_obj.split_documents(extracted_data)
    return splitted_data

def get_embedding_model():
    """
    Getting the embedding model
    """
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding