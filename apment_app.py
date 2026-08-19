# Sidebar for setup
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check if the API key is safely hidden in the Cloud Secrets
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key loaded securely from Cloud Secrets!")
    else:
        # Fallback: Ask the user to type it in if no secret is found
        api_key = st.text_input("Google Gemini API Key", type="password", help="Get this from Google AI Studio")
        st.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    st.markdown("---")
    st.header("📂 Document Upload")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    process_btn = st.button("Process Documents")
