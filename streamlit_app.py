from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000")


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def ingest_pdf(pdf_path: Path) -> dict:
    api_url = f"{get_api_base_url()}/ingest_pdf"
    response = requests.post(
        api_url,
        json={
            "pdf_path": str(pdf_path.resolve()),
            "source_id": pdf_path.name,
        }
    )
    response.raise_for_status()
    return response.json()


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and processing PDF..."):
        path = save_uploaded_pdf(uploaded)
        result = ingest_pdf(path)
    st.success(f"Successfully ingested: {result.get('ingested', path.name)}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


def query_rag(question: str) -> dict:
    api_url = f"{get_api_base_url()}/query"
    response = requests.post(
        api_url,
        json={"question": question}
    )
    response.raise_for_status()
    return response.json()


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Searching and generating answer..."):
            result = query_rag(question.strip())
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            num_contexts = result.get("num_contexts", 0)

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        st.caption(f"Used {num_contexts} context chunks")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
