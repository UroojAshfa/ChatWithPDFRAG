import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS  
from sentence_transformers import SentenceTransformer


#os.environ["HF_HOME"] = "/app/.cache/huggingface"


load_dotenv()


## If you do not have open AI key use the below Huggingface embedding
#os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
#embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    #model_kwargs={'cache_dir': "/app/.cache/huggingface"},
    encode_kwargs={'normalize_embeddings': True}
)

# Streamlit UI setup
st.set_page_config(page_title="Turbo PDF Chat", layout="wide")
st.title("⚡ Turbo Conversational RAG with PDFs")
st.write("Upload PDFs and chat with their content using Groq's fast LLMs")

def initialize_session_state():
    """Initialize all necessary session state variables"""
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

initialize_session_state()

# Groq API key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    try:
        with st.status("Processing PDFs...", expanded=True) as status:
            documents = []
            temp_dir = tempfile.TemporaryDirectory()
            
            status.update(label="Reading PDFs...")
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.processed_files:
                    continue
                
                # Save to temp file
                temp_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and split PDF
                loader = PyPDFLoader(temp_path)
                docs = loader.load_and_split()
                documents.extend(docs)
                st.session_state.processed_files.add(uploaded_file.name)
            
            if not documents:
                return None

            # Split documents
            status.update(label="Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "!", "?", ";", ", ", " "]
            )
            splits = text_splitter.split_documents(documents)
            
            # Create FAISS vector store
            status.update(label="Creating search index...")
            vectorstore = FAISS.from_documents(  # Changed to FAISS
                documents=splits,
                embedding=embeddings
            )
            
            temp_dir.cleanup()
            status.update(label="Processing complete!", state="complete")
            return vectorstore
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)

# Process files only when new files are uploaded
if uploaded_files and api_key:
    current_files = {f.name for f in uploaded_files}
    if current_files - st.session_state.processed_files:
        st.session_state.vectorstore = process_pdfs(uploaded_files)

# Chat interface
if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Gemma2-9b-It",
        temperature=0.3,
        max_tokens=1024
    )

    session_id = st.sidebar.text_input("Session ID", value="default")

    # Chat history management
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Create RAG chain only once
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Contextualization prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given chat history and a question, create a standalone question. 
            If relevant to history, rephrase it, otherwise return it unchanged. No answering."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer concisely in 1-3 sentences using ONLY the context. 
            If unsure, say 'This isn't covered in the document(s)'. Context: {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Display chat history
        for msg in get_session_history(session_id).messages:
            st.chat_message(msg.type).write(msg.content)

        # User input
        if prompt := st.chat_input("Ask about the PDFs"):
            # Add to history and display
            with st.chat_message("user"):
                st.write(prompt)

            # Get response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_text = ""
                
                # Stream the response
                for chunk in conversational_rag_chain.stream(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                ):
                    if "answer" in chunk:
                        response_text += chunk["answer"]
                        response_placeholder.markdown(response_text + "▌")
                
                response_placeholder.markdown(response_text)

else:
    st.warning("Please enter your Groq API key in the sidebar to continue")

# Debug section
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("Processed files:", st.session_state.processed_files)
    if 'vectorstore' in st.session_state and st.session_state.vectorstore:
        st.sidebar.write("Vectorstore stats:", 
                       f"{st.session_state.vectorstore.index.ntotal} chunks")  # FAISS-specific metric
    st.sidebar.write("Session store:", st.session_state.store.keys())


# import streamlit as st
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama

# # Optimize environment settings
# os.environ["OMP_NUM_THREADS"] = "6"  # Use more CPU threads
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Initialize models with Llama2 optimizations
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# llm = Ollama(
#     model="gemma",  # Use quantized 4-bit version
#     temperature=0.3,  # More deterministic outputs
#     num_ctx=2048,  # Context window size
#     num_thread=6  # Use more threads for computation
# )

# # Faster text splitting
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=2000,
#     chunk_overlap=400,
#     separators=["\n\n", "\n", ". ", "!", "?", ";", ", ", " "]
# )

# # Optimized system prompt
# SYSTEM_TEMPLATE = """
# Answer concisely in 1-3 sentences ONLY using this context. 
# If irrelevant, respond "This is not covered in the document."
# Do NOT explain your reasoning.

# Context: {context}
# Question: {input}
# Answer:"""

# @st.cache_resource
# def process_pdf(pdf_path):
#     try:
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load_and_split()
#         chunks = text_splitter.split_documents(pages)
#         return FAISS.from_documents(chunks, embeddings)
#     except Exception as e:
#         st.error(f"Processing error: {str(e)}")
#         return None

# # Streamlit UI
# st.title("Llama2 PDF Accelerator")
# st.session_state.setdefault("messages", [])
# st.session_state.setdefault("vector_store", None)

# # File upload with immediate processing
# if uploaded_file := st.file_uploader("Upload PDF", type="pdf"):
#     if not st.session_state.vector_store:
#         with st.status("Optimized Processing...", expanded=True) as status:
#             with open("temp.pdf", "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             status.update(label="Splitting PDF", state="running")
#             st.session_state.vector_store = process_pdf("temp.pdf")
#             os.remove("temp.pdf")
            
#             if st.session_state.vector_store:
#                 status.update(label="Ready for questions!", state="complete")

# # Chat interface with streaming
# if prompt := st.chat_input("Ask about the document"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     if st.session_state.vector_store:
#         with st.chat_message("assistant"):
#             retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve only 3 chunks
#             prompt_template = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
#             chain = create_retrieval_chain(
#                 retriever,
#                 create_stuff_documents_chain(llm, prompt_template)
#             )
            
#             # Stream the response
#             response = ""
#             placeholder = st.empty()
#             for chunk in chain.stream({"input": prompt}):
#                 response += chunk.get("answer", "")
#                 placeholder.markdown(response + "▌")
            
#             placeholder.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         st.warning("Upload a PDF first")
