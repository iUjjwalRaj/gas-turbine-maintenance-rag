"""Gas Turbine Maintenance Assistant - Streamlit Web Application.

A professional, industrial-engineering styled RAG interface for querying
GE GER-3620 Rev P (Heavy-Duty Gas Turbine Operating and Maintenance Considerations).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_PDF_FILENAME,
    DEFAULT_PDF_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PDF_PATH,
    USE_GEMINI,
)
from rag_pipeline import (
    RAGResponse,
    answer_question,
    download_official_pdf,
    init_or_load_vectorstore,
    is_valid_pdf,
)

load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Gas Turbine Maintenance Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial Engineering CSS Styling
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 24px 32px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 0 8px 0 !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-box p {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
    
    /* Industrial Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-navy {
        background-color: #0284c7;
        color: #ffffff;
    }
    .badge-success {
        background-color: #059669;
        color: #ffffff;
    }
    .badge-warning {
        background-color: #d97706;
        color: #ffffff;
    }
    .badge-neutral {
        background-color: #e2e8f0;
        color: #334155;
    }

    /* Cards */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Citation Table / Details */
    .citation-box {
        background: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 14px 16px;
        border-radius: 0 6px 6px 0;
        margin-top: 14px;
        font-size: 0.92rem;
    }
    .citation-label {
        font-weight: 600;
        color: #1e293b;
        margin-right: 6px;
    }
    .citation-value {
        color: #0369a1;
        font-weight: 500;
    }

    /* Unsupported Alert Box */
    .not-found-box {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        padding: 16px 20px;
        border-radius: 6px;
        color: #9f1239;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 14px;
    }

    /* Answer Text Box */
    .answer-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #0284c7;
        padding: 18px 22px;
        border-radius: 6px;
        color: #0f172a;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-top: 14px;
    }

    /* Excerpt Box */
    .excerpt-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 6px;
        padding: 12px 16px;
        font-style: italic;
        color: #334155;
        font-size: 0.9rem;
        margin-top: 10px;
    }

    /* Footer */
    .footer-text {
        color: #64748b;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 5 Required Exam Questions
SAMPLE_QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]


@st.cache_resource(show_spinner="Initializing ChromaDB Vector Index...")
def get_cached_vectorstore(pdf_file_path: str, force_reindex: bool = False):
    """Cached loader for the persistent ChromaDB collection."""
    path_obj = Path(pdf_file_path)
    return init_or_load_vectorstore(pdf_path=path_obj, force_reindex=force_reindex)


# Initialize Session State
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "active_response" not in st.session_state:
    st.session_state.active_response = None
if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = str(DEFAULT_PDF_PATH)
if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False


# Header Section
st.markdown(
    """
    <div class="header-box">
        <h1>⚙️ Gas Turbine Maintenance Assistant</h1>
        <p>Expert Retrieval-Augmented Generation (RAG) assistant for GE Heavy-Duty Gas Turbine Operating and Maintenance Considerations (GER-3620 Rev P).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Sidebar: Document Management & Indexing Status
with st.sidebar:
    st.markdown("### 📄 Document & Index Status")

    current_path = Path(st.session_state.current_pdf_path)

    # Check if default PDF exists, otherwise attempt download
    if not current_path.exists() and current_path == DEFAULT_PDF_PATH:
        with st.spinner("Downloading official GE GER-3620 Rev P PDF..."):
            try:
                download_official_pdf(target_path=current_path)
                st.success("Downloaded official GE document successfully.")
            except Exception as e:
                st.error(f"Automatic download failed: {e}")

    # File upload support
    uploaded_file = st.file_uploader(
        "Upload Custom / Updated PDF:",
        type=["pdf"],
        help="Upload the official GE GER-3620-P.pdf if not pre-downloaded.",
    )

    if uploaded_file is not None:
        save_path = DEFAULT_PDF_PATH.parent / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if is_valid_pdf(save_path):
            st.session_state.current_pdf_path = str(save_path)
            st.session_state.force_rebuild = True
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            st.error("Uploaded file is not a valid PDF.")

    # Re-index / Force refresh button
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        st.session_state.force_rebuild = True

    # Vectorstore loading
    vectorstore = None
    stats = None
    pdf_ready = is_valid_pdf(Path(st.session_state.current_pdf_path))

    if pdf_ready:
        try:
            vectorstore, stats = get_cached_vectorstore(
                st.session_state.current_pdf_path,
                force_reindex=st.session_state.force_rebuild,
            )
            # Reset force rebuild flag after successful load
            st.session_state.force_rebuild = False

            st.markdown(
                f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:12px; margin-top:12px;">
                    <div style="font-size:0.85rem; margin-bottom:4px;"><b>Document:</b> <code>{stats.get('filename')}</code></div>
                    <div style="font-size:0.85rem; margin-bottom:4px;"><b>Pages:</b> {stats.get('page_count')}</div>
                    <div style="font-size:0.85rem; margin-bottom:4px;"><b>Chunks:</b> {stats.get('chunk_count')}</div>
                    <div style="font-size:0.85rem; margin-bottom:4px;"><b>Cache Status:</b> <span class="badge badge-success">{'Reused Index' if stats.get('reused') else 'Built Fresh'}</span></div>
                    <div style="font-size:0.85rem;"><b>Embedding:</b> <span class="badge badge-neutral">all-MiniLM-L6-v2</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as err:
            st.error(f"Error initializing vector index: {err}")
    else:
        st.warning("⚠️ Document not found. Please download or upload GER-3620-P.pdf.")

    st.markdown("---")
    st.markdown("### ⚙️ Engine Mode")
    has_gemini = bool(GEMINI_API_KEY and USE_GEMINI)
    engine_badge = (
        f"<span class='badge badge-success'>Gemini Flash ({GEMINI_MODEL})</span>"
        if has_gemini
        else "<span class='badge badge-neutral'>Local Extractive (Offline Fallback)</span>"
    )
    st.markdown(
        f"""
        - **Generator:** {engine_badge}
        - **Retrieval:** Local sentence-transformers (`all-MiniLM-L6-v2`)
        - **Vector DB:** Persistent ChromaDB (`chroma_db/`)
        - **Grounding Rule:** Strict evidence gating; returns **`Not found`** on unsupported queries.
        """,
        unsafe_allow_html=True,
    )


# Main Interface Layout
col_main, col_examples = st.columns([7, 5])

with col_examples:
    st.markdown("#### 📋 Required Exam Questions")
    st.caption("Click any question to load and run against the indexed GE document:")

    for idx, sample_q in enumerate(SAMPLE_QUESTIONS, 1):
        if st.button(f"{idx}. {sample_q}", key=f"sample_{idx}", use_container_width=True):
            st.session_state.query_input = sample_q
            if vectorstore is not None:
                with st.spinner("Retrieving evidence and evaluating..."):
                    st.session_state.active_response = answer_question(
                        sample_q, vectorstore
                    )
            st.rerun()

with col_main:
    st.markdown("#### 🔍 Ask a Question")

    # Form for question submission
    with st.form("qa_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter maintenance or operational question:",
            value=st.session_state.query_input,
            placeholder="e.g. What are the three main types of inspection?",
        )
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            submit_btn = st.form_submit_button("⚡ Ask Assistant", use_container_width=True)
        with col_btn2:
            clear_btn = st.form_submit_button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.query_input = ""
        st.session_state.active_response = None
        st.rerun()

    if submit_btn and user_query.strip():
        st.session_state.query_input = user_query.strip()
        if vectorstore is None:
            st.error("Vector store is not initialized. Please ensure the PDF is indexed.")
        else:
            with st.spinner("Analyzing document passages and validating evidence..."):
                response = answer_question(user_query.strip(), vectorstore)
                st.session_state.active_response = response

    # Display Active Response
    if st.session_state.active_response is not None:
        resp: RAGResponse = st.session_state.active_response

        st.markdown("---")
        st.markdown("### 💬 Answer")

        if not resp.is_supported or resp.answer == "Not found":
            st.markdown(
                """
                <div class="not-found-box">
                    ⚠️ Not found
                    <div style="font-weight:400; font-size:0.88rem; margin-top:6px; color:#475569;">
                        The requested information is not supported by the document (GE GER-3620 Rev P).
                        In strict compliance with RAG verification rules, external knowledge and speculation are prohibited.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="answer-box">
                    {resp.answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Source Citation Panel
            st.markdown("### 📌 Supporting Source Citation")
            printed_display = (
                f"Page {resp.printed_page}"
                if resp.printed_page and resp.printed_page != "Unavailable"
                else "N/A"
            )
            st.markdown(
                f"""
                <div class="citation-box">
                    <div><span class="citation-label">Document:</span> <span class="citation-value">{resp.source}</span></div>
                    <div><span class="citation-label">Physical PDF Page:</span> <span class="citation-value">Page {resp.page} of 60</span> | <span class="citation-label">Printed Page:</span> <span class="citation-value">{printed_display}</span></div>
                    <div><span class="citation-label">Section Heading:</span> <span class="citation-value">{resp.section}</span></div>
                    <div class="excerpt-box">
                        <strong>Source Excerpt:</strong><br>
                        "{resp.excerpt}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Retrieval / Debug Panel (Collapsible)
        if resp.retrieved_chunks:
            with st.expander("🔍 Inspect Retrieved Passages & Relevance Scores (Debug Panel)"):
                st.caption(
                    "Top passages retrieved from ChromaDB with cosine distance scores (lower distance = higher similarity):"
                )
                for i, chunk in enumerate(resp.retrieved_chunks, 1):
                    col_meta1, col_meta2 = st.columns([3, 1])
                    with col_meta1:
                        st.markdown(
                            f"**Passage #{i}** — *{chunk['source']}* (PDF P.{chunk['page']}, Printed P.{chunk['printed_page']}) | Section: `{chunk['section']}`"
                        )
                    with col_meta2:
                        st.markdown(f"**Distance:** `{chunk['score']}`")
                    st.text_area(
                        f"Chunk #{i} Content",
                        chunk["content"],
                        height=110,
                        key=f"chunk_view_{i}",
                        disabled=True,
                    )
                    st.markdown("---")


# Footer Section
st.markdown(
    """
    <div class="footer-text">
        Gas Turbine Maintenance Assistant | Based on GE Vernova Technical Document <code>GER-3620 Rev P</code> (Heavy-Duty Gas Turbine Operating and Maintenance Considerations).
    </div>
    """,
    unsafe_allow_html=True,
)
