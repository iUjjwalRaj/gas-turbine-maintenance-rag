"""Gas Turbine Maintenance Assistant - Cinematic 3D Streamlit Web Application.

A full-screen 3D industrial engineering experience featuring an interactive
Three.js cutaway gas turbine and a floating glassmorphic AI chat panel for querying
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
from components.turbine_viewer import turbine_viewer

load_dotenv()

# Streamlit Page Configuration - Fullscreen wide layout, sidebar collapsed
st.set_page_config(
    page_title="Turbine AI — Gas Turbine Maintenance Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 5 Required Exam Questions
SAMPLE_QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]


@st.cache_resource(show_spinner="Initializing Vector Database...")
def get_cached_vectorstore(pdf_file_path: str, force_reindex: bool = False):
    """Cached loader for the persistent ChromaDB collection."""
    path_obj = Path(pdf_file_path)
    return init_or_load_vectorstore(pdf_path=path_obj, force_reindex=force_reindex)


# Initialize Session State
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "active_response" not in st.session_state:
    st.session_state.active_response = None
if "active_query_text" not in st.session_state:
    st.session_state.active_query_text = ""
if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = str(DEFAULT_PDF_PATH)
if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False
if "turbine_selected" not in st.session_state:
    st.session_state.turbine_selected = None
if "turbine_last_ts" not in st.session_state:
    st.session_state.turbine_last_ts = None


# Core Cinematic Glassmorphism Styling
st.markdown(
    """
    <style>
    /* 1. Global Reset & Near-Black Palette */
    html, body, .stApp {
        background-color: #080B10 !important;
        color: #E6EAF0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        overflow: hidden !important;
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide Default Streamlit Chrome */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }

    /* Container zeroing to allow full-bleed 3D viewport */
    .block-container {
        padding: 0 !important;
        max-width: 100vw !important;
        margin: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;
        position: relative !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    /* 2. Fullscreen 3D Turbine Viewport Container */
    .st-key-turbine_fullscreen_bg,
    div:has(> #turbine-fullscreen-anchor) {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 1 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        pointer-events: auto !important;
    }
    .st-key-turbine_fullscreen_bg iframe,
    div:has(> #turbine-fullscreen-anchor) iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
    }

    /* 3. Top Header Brand Bar (Floating Top-Left) */
    .brand-bar {
        position: fixed;
        top: 20px;
        left: 24px;
        z-index: 95;
        display: flex;
        align-items: center;
        gap: 12px;
        pointer-events: none;
    }
    .brand-pill {
        background: rgba(16, 22, 30, 0.82);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #E6EAF0;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 16px rgba(58, 160, 255, 0.15);
    }
    .brand-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3AA0FF;
        box-shadow: 0 0 10px #3AA0FF;
        animation: pulse-dot 2.5s infinite ease-in-out;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }

    /* 4. Top-Right System & Document Status Expander Container */
    .st-key-top_status_panel,
    div:has(> #top-status-anchor) {
        position: fixed !important;
        top: 20px !important;
        right: 24px !important;
        z-index: 95 !important;
        width: 330px !important;
        pointer-events: auto !important;
    }
    .st-key-top_status_panel .streamlit-expanderHeader,
    div:has(> #top-status-anchor) .streamlit-expanderHeader {
        background: rgba(16, 22, 30, 0.82) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        color: #E6EAF0 !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }
    .st-key-top_status_panel .streamlit-expanderContent,
    div:has(> #top-status-anchor) .streamlit-expanderContent {
        background: rgba(14, 18, 24, 0.94) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        margin-top: 6px !important;
        padding: 14px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6) !important;
    }

    /* 5. Floating Glassmorphism Chat Panel (Lower Left) */
    .st-key-floating_chat_panel,
    div:has(> #floating-chat-anchor) {
        position: fixed !important;
        bottom: 24px !important;
        left: 24px !important;
        width: 410px !important;
        max-width: calc(100vw - 48px) !important;
        max-height: calc(100vh - 84px) !important;
        z-index: 100 !important;
        background: rgba(13, 17, 23, 0.82) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65), 0 0 24px rgba(58, 160, 255, 0.12) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        pointer-events: auto !important;
    }

    /* Chat Panel Header */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 12px;
        margin-bottom: 12px;
    }
    .chat-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .chat-status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #38bdf8;
    }
    .status-emerald {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* User Query Display */
    .user-query-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .user-query-card b {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Supported Answer Card */
    .answer-card {
        background: rgba(18, 24, 34, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-left: 3px solid #3AA0FF;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #F1F5F9;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    /* Strict Not-Found Warning Card */
    .not-found-card {
        background: rgba(45, 12, 18, 0.65);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-left: 3px solid #f43f5e;
        border-radius: 10px;
        padding: 14px 16px;
        color: #fda4af;
        margin-bottom: 12px;
    }
    .not-found-badge {
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        color: #ff4d6d;
    }
    .not-found-desc {
        font-size: 0.82rem;
        color: #fca5a5;
        line-height: 1.45;
    }

    /* Citation Card Inside Expander */
    .cite-block {
        font-size: 0.82rem;
        line-height: 1.5;
        color: #cbd5e1;
    }
    .cite-block strong {
        color: #38bdf8;
    }
    .cite-excerpt {
        background: rgba(10, 14, 20, 0.6);
        border-left: 2px solid #38bdf8;
        border-radius: 4px;
        padding: 8px 10px;
        margin-top: 8px;
        font-style: italic;
        color: #94a3b8;
        font-size: 0.80rem;
    }

    /* Hints and Footers */
    .chat-hint {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 8px;
        text-align: center;
        letter-spacing: 0.02em;
    }

    /* Input & Button Overrides for Modern Dark Glass */
    .stTextInput input {
        background: rgba(22, 28, 38, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
        font-size: 0.90rem !important;
        padding: 9px 12px !important;
    }
    .stTextInput input:focus {
        border-color: #3AA0FF !important;
        box-shadow: 0 0 12px rgba(58, 160, 255, 0.35) !important;
    }

    .stButton > button {
        background: rgba(24, 32, 44, 0.85) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        padding: 6px 12px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(36, 48, 64, 0.95) !important;
        border-color: #3AA0FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 12px rgba(58, 160, 255, 0.25) !important;
    }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(58, 160, 255, 0.4) !important;
        border-radius: 9px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35) !important;
    }
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
        box-shadow: 0 0 16px rgba(58, 160, 255, 0.5) !important;
    }

    /* Slim Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(58, 160, 255, 0.5);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .st-key-floating_chat_panel,
        div:has(> #floating-chat-anchor) {
            left: 12px !important;
            bottom: 12px !important;
            width: calc(100vw - 24px) !important;
            max-height: 50vh !important;
        }
        .st-key-top_status_panel,
        div:has(> #top-status-anchor) {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Document & Indexing Initialization
current_path = Path(st.session_state.current_pdf_path)
if not current_path.exists() and current_path == DEFAULT_PDF_PATH:
    try:
        download_official_pdf(target_path=current_path)
    except Exception as e:
        st.error(f"Automatic download failed: {e}")

pdf_ready = is_valid_pdf(current_path)
vectorstore = None
stats = None

if pdf_ready:
    try:
        vectorstore, stats = get_cached_vectorstore(
            st.session_state.current_pdf_path,
            force_reindex=st.session_state.force_rebuild,
        )
        st.session_state.force_rebuild = False
    except Exception as err:
        st.error(f"Error initializing vector index: {err}")


# =========================================================================
# LAYER 1: Full-Screen 3D Turbine Viewer (Viewport Background)
# =========================================================================
with st.container(key="turbine_fullscreen_bg"):
    st.markdown('<div id="turbine-fullscreen-anchor"></div>', unsafe_allow_html=True)
    viewer_event = turbine_viewer(
        selected_id=st.session_state.get("turbine_selected"),
        height=950,
        key="turbine_3d_viewer",
    )
    if viewer_event and isinstance(viewer_event, dict) and viewer_event.get("ts") != st.session_state.get("turbine_last_ts"):
        st.session_state.turbine_last_ts = viewer_event.get("ts")
        if viewer_event.get("type") == "select":
            st.session_state.turbine_selected = viewer_event.get("componentId")
        elif viewer_event.get("type") == "ask":
            st.session_state.query_input = viewer_event.get("question", "")
            st.rerun()


# =========================================================================
# LAYER 2: Floating Brand Bar (Top Left)
# =========================================================================
st.markdown(
    """
    <div class="brand-bar">
        <div class="brand-pill">
            <span class="brand-dot"></span>
            <span>GE GER-3620 Rev P &bull; Gas Turbine Assistant</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# LAYER 3: Floating Document & System Status (Top Right Dropdown Expander)
# =========================================================================
with st.container(key="top_status_panel"):
    st.markdown('<div id="top-status-anchor"></div>', unsafe_allow_html=True)
    with st.expander("⚙️ Document & System Status", expanded=False):
        if stats:
            has_gemini = bool(GEMINI_API_KEY and USE_GEMINI)
            engine_text = f"Gemini Flash ({GEMINI_MODEL})" if has_gemini else "Local Extractive (Offline)"
            st.markdown(
                f"""
                <div style="font-size:0.80rem; line-height:1.6; color:#cbd5e1;">
                    <div><b>Document:</b> <code>{stats.get('filename')}</code></div>
                    <div><b>Pages:</b> {stats.get('page_count')} &bull; <b>Chunks:</b> {stats.get('chunk_count')}</div>
                    <div><b>Index:</b> {'Reused Cache' if stats.get('reused') else 'Built Fresh'}</div>
                    <div><b>Generator:</b> <span style="color:#38bdf8;">{engine_text}</span></div>
                    <div><b>Retrieval:</b> <code>all-MiniLM-L6-v2</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Index not loaded yet.")

        # Re-index button
        if st.button("🔄 Rebuild Vector Index", key="btn_reindex", use_container_width=True):
            st.session_state.force_rebuild = True
            st.rerun()

        # PDF Uploader
        uploaded_pdf = st.file_uploader(
            "Upload Custom PDF:",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_pdf is not None:
            save_path = DEFAULT_PDF_PATH.parent / uploaded_pdf.name
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            if is_valid_pdf(save_path):
                st.session_state.current_pdf_path = str(save_path)
                st.session_state.force_rebuild = True
                st.success(f"Loaded: {uploaded_pdf.name}")
                st.rerun()


# =========================================================================
# LAYER 4: Floating Glassmorphism Chat Panel (Lower Left)
# =========================================================================
with st.container(key="floating_chat_panel"):
    st.markdown('<div id="floating-chat-anchor"></div>', unsafe_allow_html=True)

    # Chat Header
    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-title">
                <span>⚙️</span> TURBINE AI
            </div>
            <div class="chat-status">
                <span class="status-emerald"></span>
                <span>READY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 Required Exam Questions in Compact Collapsible Area
    with st.expander("⚡ Exam Sample Questions (5)", expanded=False):
        for idx, sample_q in enumerate(SAMPLE_QUESTIONS, 1):
            if st.button(f"{idx}. {sample_q}", key=f"sample_q_{idx}", use_container_width=True):
                st.session_state.query_input = sample_q
                st.session_state.active_query_text = sample_q
                if vectorstore is not None:
                    with st.spinner("Analyzing manual evidence..."):
                        resp = answer_question(sample_q, vectorstore)
                        st.session_state.active_response = resp
                st.rerun()

    # Active Conversation & Answer Display Area
    if st.session_state.active_response is not None:
        resp: RAGResponse = st.session_state.active_response

        # Show active query
        if st.session_state.active_query_text:
            st.markdown(
                f"""
                <div class="user-query-card">
                    <b>Q:</b> {st.session_state.active_query_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Unsupported / Not found
        if not resp.is_supported or resp.answer == "Not found":
            st.markdown(
                """
                <div class="not-found-card">
                    <div class="not-found-badge">⚠️ Not found</div>
                    <div class="not-found-desc">
                        The requested information is not supported by the document (GE GER-3620 Rev P).
                        Strict evidence validation active: ungrounded answers and hallucinations are prevented.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Supported Answer
            st.markdown(
                f"""
                <div class="answer-card">
                    {resp.answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Collapsible Source Citation
            printed_disp = (
                f"Page {resp.printed_page}"
                if resp.printed_page and resp.printed_page != "Unavailable"
                else "N/A"
            )
            with st.expander("📌 Supporting Source Citation", expanded=True):
                st.markdown(
                    f"""
                    <div class="cite-block">
                        <div><strong>Document:</strong> {resp.source}</div>
                        <div><strong>Physical PDF:</strong> Page {resp.page} of 60 | <strong>Printed Page:</strong> {printed_disp}</div>
                        <div><strong>Section:</strong> {resp.section}</div>
                        <div class="cite-excerpt">"{resp.excerpt}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Debug retrieved chunks (collapsible)
            if resp.retrieved_chunks:
                with st.expander("🔍 ChromaDB Retrieval Passages", expanded=False):
                    for i, chunk in enumerate(resp.retrieved_chunks, 1):
                        st.caption(
                            f"**#{i}** P.{chunk['page']} (Score: `{chunk['score']}`): `{chunk['section']}`"
                        )
                        st.text_area(
                            f"Passage {i}",
                            chunk["content"],
                            height=75,
                            key=f"chunk_box_{i}",
                            disabled=True,
                        )

    else:
        # Default empty state greeting
        st.markdown(
            """
            <div style="background:rgba(20,26,36,0.5); border:1px dashed rgba(255,255,255,0.08); border-radius:10px; padding:12px; margin-bottom:12px; font-size:0.83rem; color:#94a3b8; text-align:center;">
                💡 Ask any maintenance or operational question, or select an exam question from the menu above.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Question Input Form
    with st.form("chat_qa_form", clear_on_submit=False):
        user_input = st.text_input(
            "Question:",
            value=st.session_state.query_input,
            placeholder="Ask about inspection, maintenance factor...",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            submit_ask = st.form_submit_button("⚡ Ask Assistant", use_container_width=True)
        with col_clear:
            clear_btn = st.form_submit_button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.query_input = ""
        st.session_state.active_query_text = ""
        st.session_state.active_response = None
        st.rerun()

    if submit_ask and user_input.strip():
        st.session_state.query_input = user_input.strip()
        st.session_state.active_query_text = user_input.strip()
        if vectorstore is None:
            st.error("Vector database is not ready. Please verify PDF index.")
        else:
            with st.spinner("Retrieving evidence & validating..."):
                answer = answer_question(user_input.strip(), vectorstore)
                st.session_state.active_response = answer
            st.rerun()

    # Footer Hint
    st.markdown(
        """
        <div class="chat-hint">
            Ask about this turbine &bull; Strict grounding in GE GER-3620 Rev P
        </div>
        """,
        unsafe_allow_html=True,
    )
