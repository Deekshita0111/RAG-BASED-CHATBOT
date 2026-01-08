# import os
# import google.generativeai as genai
# from langchain.vectorstores import FAISS  # This will be the vector database
# from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # This for chunking
# from pypdf import PdfReader
# import faiss
# import streamlit as st
# from pdfextractor import text_extractor_pdf

# from dotenv import load_dotenv
# load_dotenv()


# # Create the main page
# st.title(':green[RAG Based CHATBOT]')
# tips = '''Follow the steps to use this application:
# * Upload your pdf document in sidebar.
# * Write your query and start chatting with the bot.'''
# st.subheader(tips)

# # Load PDF in Side Bar
# st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF Only)]')
# file_uploaded = st.sidebar.file_uploader('Upload File')
# if file_uploaded:
#     file_text = text_extractor_pdf(file_uploaded)
#     # Step 1: Configure the models

#     # Configure LLM
#     key = os.getenv('GOOGLE_API_KEY')
#     genai.configure(api_key=key)
#     llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

#     # Configure Embedding Model
#     embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

#     # Step 2 : Chunking (Create Chunks)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap = 200)
#     chunks = splitter.split_text(file_text)

#     # Step 3: Create FAISS Vector Store
#     vector_store = FAISS.from_texts(chunks,embedding_model)

#     # Step 4: Configure retriever
#     retriever = vector_store.as_retriever(search_kwargs={"k":3})

#     # Lets create a function that takes query and return the generated text
#     def generate_response(query):
#         # Step 6 : Retrieval (R)
#         retrived_docs = retriever.get_relevant_documents(query=query)
#         context = ' '.join([doc.page_content for doc in retrived_docs])

#         # Step 7: Write a Augmeneted prompt (A)
#         prompt = f'''You are a helpful assitant using RAG
#         Here is the context = {context}
#         The query asked by user is as follows = {query}'''

#         # Step 9: Generation (G)
#         content = llm_model.generate_content(prompt)
#         return content.text

    
#     # Lets create a chatbot in order to start the converstaion
#     # Initialize chat if there is no history
#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     # Display the History
#     for msg in st.session_state.history:
#         if msg['role'] == 'user':
#             st.write(f':green[User:] :blue[{msg['text']}]')
#         else:
#             st.write(f':orange[Chatbot:]  {msg['text']}')

#     # Input from the user (Using Steamlit Form)
#     with st.form('Chat Form',clear_on_submit=True):
#         user_input = st.text_input('Enter Your Text Here: ')
#         send = st.form_submit_button('Send')
    
#     # Start the converstaion and append the output and query in history
#     if user_input and send:

#         st.session_state.history.append({"role":'user',"text":user_input})

#         model_output = generate_response(user_input)

#         st.session_state.history.append({'role':'chatbot','text':model_output})

#         st.rerun()

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)



# ---------------- GEMINI CONFIG ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     st.error("❌ GOOGLE_API_KEY not found in environment variables")
#     st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_resource
def load_llm():
    return genai.GenerativeModel('gemini-flash-lite-latest')

llm_model = load_llm()

# ---------------- UI ----------------
st.title("📄 RAG Based PDF Chatbot")
st.markdown("""
**How to use:**
1. Upload a PDF document  
2. Ask questions based on the document  
3. Get AI-generated answers using Retrieval-Augmented Generation  
""")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📤 Upload PDF Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ---------------- PDF PROCESSING ----------------
if uploaded_file:

    with st.spinner("📚 Reading and indexing document..."):

        # Save PDF temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        vector_store = FAISS.from_documents(chunks, embedding_model)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    st.success("✅ Document indexed successfully!")

    # ---------------- CHAT STATE ----------------
    if "history" not in st.session_state:
        st.session_state.history = []

    # ---------------- DISPLAY CHAT ----------------
    st.markdown("## 💬 Chat")

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 User:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

    # ---------------- QUERY INPUT ----------------
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question from the PDF:")
        send = st.form_submit_button("Send")

    # ---------------- RAG PIPELINE ----------------
    if send and user_query:

        st.session_state.history.append({
            "role": "user",
            "content": user_query
        })

        with st.spinner("🔍 Retrieving relevant context..."):
            retrieved_docs = retriever.invoke(user_query)
            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

        prompt = f"""
You are a helpful AI assistant using Retrieval-Augmented Generation (RAG).

Use ONLY the context provided below to answer the question.

Context:
{context}

Question:
{user_query}

Answer clearly and concisely.
"""

        response = llm_model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )

        st.session_state.history.append({
            "role": "assistant",
            "content": response.text
        })

        st.rerun()


        




