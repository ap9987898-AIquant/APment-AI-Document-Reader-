import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Page Configuration
st.set_page_config(page_title="APment - PDF AI Agent", page_icon="📄", layout="wide")

# Custom CSS for better presentation UI
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #000000; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📄 APment: Your PDF AI")
st.write("Upload a document, process it, and ask questions. Built with LangChain and Streamlit.")

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get this from Google AI Studio")
    st.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    st.markdown("---")
    st.header("📂 Document Upload")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    process_btn = st.button("Process Documents")

# Core Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    # Splits the document into small pieces so the AI can read them efficiently
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    # Uses local HuggingFace embeddings. This saves time and API costs!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are APment, an intelligent assistant. Answer the question carefully based ONLY on the provided context. 
    If the answer is not in the provided context, state clearly: "The answer is not available in the uploaded document." 
    Do not hallucinate or make up information.

    Context:
    {context}?

    Question: 
    {question}

    Answer:
    """
    # Uses Gemini for the final answer generation
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Using modern LangChain LCEL to completely bypass the broken legacy module
    return prompt | model | StrOutputParser()

# App Logic: Processing
if process_btn:
    if not api_key:
        st.error("⚠️ Please enter your API Key in the sidebar.")
    elif not pdf_docs:
        st.error("⚠️ Please upload at least one PDF.")
    else:
        with st.spinner("APment is processing your PDF... (Embedding chunks locally)"):
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Documents processed and stored in local vector database successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")

# App Logic: Chat Interface
user_question = st.chat_input("Ask APment a question about your PDF...")

if user_question:
    # Display user question
    with st.chat_message("user"):
        st.write(user_question)
        
    if not api_key:
        st.error("⚠️ API Key required to generate answers.")
    elif not os.path.exists("faiss_index"):
        st.error("⚠️ Please upload and process a PDF first.")
    else:
        with st.spinner("APment is searching the document and generating an answer..."):
            try:
                # Load local database
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                
                # Find relevant paragraphs
                docs = vector_store.similarity_search(user_question)
                
                # Combine the retrieved document chunks into a single text block
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate Answer using modern LCEL
                chain = get_conversational_chain()
                response_text = chain.invoke({"context": context_text, "question": user_question})
                
                # Display Answer
                with st.chat_message("assistant"):
                    st.write(response_text)
            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")










