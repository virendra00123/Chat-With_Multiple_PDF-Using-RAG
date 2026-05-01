import os
import streamlit as st
from pdf_parser import extract_text, chunk_text
from embedder import embed_chunks, store_embeddings
from retriever import search_similar_chunks
from llm_interface import generate_answer

st.set_page_config(page_title="PDF Chat with Gemini", layout="wide")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(pdf_file):
    # Save uploaded PDF to temp directory
    temp_path = os.path.join("temp", pdf_file.name)
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())

    # Extract text and chunk
    text = extract_text(temp_path)
    chunks, metadata = chunk_text(text, filename=pdf_file.name)
    for i in range(len(chunks)):
        metadata[i]["text"] = chunks[i]
    return chunks, metadata

def build_index_from_uploaded_pdfs(uploaded_files):
    all_chunks = []
    all_metadata = []
    for pdf_file in uploaded_files:
        chunks, metadata = process_pdf(pdf_file)
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)
    embeddings = embed_chunks(all_chunks)
    store_embeddings(embeddings, all_metadata)

def chat_with_pdf(question):
    top_chunks, top_meta = search_similar_chunks(question, k=3)
    context = "\n".join(top_chunks)
    answer = generate_answer(context, question)
    return answer, top_meta

# --- UI ---

st.title("📄 Chat with Your PDFs (Gemini Flash 1.5)")

st.markdown("Upload one or more PDF files below, and then ask questions based on their content.")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🔍 Process PDFs"):
        with st.spinner("Processing and indexing PDFs..."):
            os.makedirs("temp", exist_ok=True)
            build_index_from_uploaded_pdfs(uploaded_files)
        st.success("✅ PDFs processed. You can now ask questions below!")

    if "vector_db_built" not in st.session_state:
        st.session_state.vector_db_built = True

if st.session_state.get("vector_db_built", False):
    user_input = st.text_input("Ask a question about the uploaded PDFs:")

    if user_input:
        with st.spinner("Thinking..."):
            answer, sources = chat_with_pdf(user_input)

        st.session_state.chat_history.append((user_input, answer, sources))

    # Display chat history
    for i, (q, a, meta) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Gemini Flash:** {a}")
        st.markdown("**Sources:**")
        for m in meta:
            st.markdown(f"- {m['source']} [chunk {m['chunk_index']}]")
        st.markdown("---")
