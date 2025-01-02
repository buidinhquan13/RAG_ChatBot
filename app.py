import streamlit as st
import os
import shutil
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever
import ollama


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "gemma2"
EMBEDDING_MODEL = "bge-m3"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

#Pull model
ollama.pull(EMBEDDING_MODEL)
ollama.pull(MODEL_NAME)

def ingest_document(file_or_url, content_type):
    """Load PDF or URL documents."""
    try:
        if content_type == "PDF":
            loader = UnstructuredPDFLoader(file_path=file_or_url)
        elif content_type == "Website URL":
            loader = WebBaseLoader(file_or_url)
        data = loader.load()
        logging.info(f"{content_type} loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load {content_type}: {str(e)}")
        st.error(f"Failed to load the {content_type}. Please try again.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_parent_splitter = RecursiveCharacterTextSplitter(chunk_size=700, 
                                                   chunk_overlap=150,
                                                   length_function=len,
                                                   separators=["\n\n\n\n\n\n",
                                                               "\n\n\n\n\n",
                                                                "\n\n\n\n",
                                                                '                                                 ',
                                                               "\u200b"])
    
    text_child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                                   chunk_overlap=70,
                                                   length_function=len,
                                                   separators=["\n\n",
                                                               "\u200b"])
    parent_chunks = text_parent_splitter.split_documents(documents)
    child_chunks = text_child_splitter.split_documents(parent_chunks)
    chunks = parent_chunks + child_chunks
    logging.info("Documents split into chunks.")
    return chunks

def delete_vector_db():
    """Delete the existing vector database."""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        logging.info("Deleted existing vector database.")

def load_vector_db(file_or_url, content_type):
    """Load or create the vector database."""
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the uploaded document
        data = ingest_document(file_or_url, content_type)
        if data is None:
            return None
        chunks = split_documents(data)
        vector_db = Chroma.from_documents(
            documents= chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db



def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Bạn là hệ thống chatbot RAG đang hỗ trợ người dùng trả lời câu hỏi dựa trên dữ liệu đã cung cấp.
        Nhiệm vụ bạn là tạo nhiều góc nhìn khác nhau cho câu hỏi, hệ thống RAG khắc phục một số hạn chế của phương pháp tìm kiếm dựa trên khoảng cách tương đồng Cosine Similarity.
        Bạn hãy phân tích câu hỏi của người dùng và tạo ra 5 phiên bản khác nhau dựa trên câu hỏi của người dùng. Hãy liệt kê ra 5 câu hỏi đó và không trả lời gì thêm.
        Câu hỏi của người dùng: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    template = """Bạn là hệ thống chatbot RAG đang hỗ trợ người dùng trả lời câu hỏi dựa trên dữ liệu đã cung cấp. 
    Hãy dựa vào các thông tin dưới đầy để trả lời câu hỏi của người dùng một cách chuyên nghiệp: {context}
    Câu hỏi của người dùng: {question}
    Nếu người dùng hỏi về thông tin không tồn tại trong dữ liệu được cung cấp, hãy trả lời rằng không tìm thấy thông tin.
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

def main():
    st.title("Chatbot Assistant 🤖")
    st.caption("Upload your document or enter a URL and ask any question. The chatbot will try to answer using the provided document.")
    
    content_type = st.sidebar.selectbox("Select Content Type", ["PDF", "Website URL"])
    file_or_url = None

    if content_type == "PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file:
            file_or_url = f"./uploaded_{uploaded_file.name}"
            with open(file_or_url, "wb") as f:
                f.write(uploaded_file.read())
    elif content_type == "Website URL":
        url = st.sidebar.text_input("Enter a Website URL")
        if url.strip():
            file_or_url = url

    if st.sidebar.button("Reset Database"):
        delete_vector_db()
        st.session_state.pop("vector_db", None)
        st.session_state.pop("messages", None)
        st.success("Database reset successfully. Please upload a new document or enter a new URL.")

    if file_or_url:
        llm = ChatOllama(model=MODEL_NAME)

        if "vector_db" not in st.session_state:
            vector_db = load_vector_db(file_or_url, content_type)
            if vector_db is None:
                st.error("Failed to load or create the vector database.")
                return
            st.session_state.vector_db = vector_db
        else:
            vector_db = st.session_state.vector_db

    
        retriever = create_retriever(vector_db, llm)
        chain = create_chain(retriever, llm)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter your question:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
                

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    response = chain.invoke(input=prompt)
                    full_response = response
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.info("Please upload a PDF file or enter a URL to get started.")
        st.warning("Câu trả lời của chatbot có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.")

if __name__ == "__main__":
    main()
