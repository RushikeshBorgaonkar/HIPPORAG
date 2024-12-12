from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


def load_pdf_with_langchain(pdf_path, chunk_size=1000, overlap=50):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)
        
        # Use RecursiveCharacterTextSplitter to split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = text_splitter.split_text(text)
        print("CHUNKS : ",chunks)
        return chunks
    except Exception as e:
        print(f"Error reading PDF with LangChain: {e}")
        return []