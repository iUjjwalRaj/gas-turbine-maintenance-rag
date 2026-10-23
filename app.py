"""Gas Turbine Maintenance Assistant - Cinematic 3D Streamlit Web Application.

A full-screen 3D industrial engineering experience featuring an interactive
Three.js cutaway gas turbine and a floating glassmorphic AI chat panel for querying
GE GER-3620 Rev P (Heavy-Duty Gas Turbine Operating and Maintenance Considerations).
"""

from __future__ import annotations

import html
import os
import re
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


def format_answer_html(raw_answer: str) -> str:
    """Format RAG answer text into clear structured typography with lists and paragraphs.

    Strictly preserves all numbers, facts, and verbatim wording without truncation.
    """
    if not raw_answer:
        return ""
    text = raw_answer.strip()
    if text == "Not found":
        return text

    # Pattern 1: Embedded numbered list like (1) Item A, (2) Item B, and (3) Item C
    paren_pattern = r"(?:\s*\((\d+)\)\s*)(.+?)(?=(?:\s*\(\d+\)|$))"
    intro_match = re.match(r"^(.*?)(?:\:\s*\(1\)|\s+\(1\))", text, flags=re.DOTALL)
    if intro_match and "(2)" in text:
        intro = intro_match.group(1).strip()
        if not intro.endswith(":"):
            intro += ":"
        items = re.findall(paren_pattern, text[len(intro_match.group(1)):], flags=re.DOTALL)
        if items:
            out = f'<p class="answer-lead">{html.escape(intro)}</p><ul class="answer-list">'
            for num, item in items:
                clean_item = item.strip()
                clean_item = re.sub(r"^(?:and|or)\s+", "", clean_item, flags=re.IGNORECASE)
                clean_item = re.sub(r"(?:,\s*(?:and|or)|[,;])\s*$", "", clean_item, flags=re.IGNORECASE)
                out += (
                    f'<li><span class="list-badge">{html.escape(num)}</span>'
                    f'<span class="list-text">{html.escape(clean_item)}</span></li>'
                )
            out += "</ul>"
            return out

    # Pattern 2: Multi-line list items (e.g., 1. Item or - Item)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1 and all(re.match(r"^(?:[-*•]|\d+[\.\)])\s+", l) for l in lines[1:]):
        intro = lines[0]
        out = f'<p class="answer-lead">{html.escape(intro)}</p><ul class="answer-list">'
        for line in lines[1:]:
            m = re.match(r"^(?:[-*•]|\s*(\d+)[\.\)])\s+(.*)", line)
            num = m.group(1) if (m and m.group(1)) else "•"
            content = m.group(2) if m else line
            out += (
                f'<li><span class="list-badge">{html.escape(num)}</span>'
                f'<span class="list-text">{html.escape(content)}</span></li>'
            )
        out += "</ul>"
        return out

    # Pattern 3: Standard prose - split on double newline or single newline
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    out_paras = []
    for p in paragraphs:
        out_paras.append(f'<p class="answer-paragraph">{html.escape(p)}</p>')
    return "".join(out_paras)


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
    .st-key-top_status_panel [data-testid="stExpander"] summary,
    .st-key-top_status_panel .stExpander summary,
    .st-key-top_status_panel .streamlit-expanderHeader,
    div:has(> #top-status-anchor) [data-testid="stExpander"] summary,
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
    .st-key-top_status_panel [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    .st-key-top_status_panel .stExpander [data-testid="stExpanderDetails"],
    .st-key-top_status_panel .streamlit-expanderContent,
    div:has(> #top-status-anchor) [data-testid="stExpander"] [data-testid="stExpanderDetails"],
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

    /* 5. Floating Glassmorphism Chat Panel (Left Side Viewport) */
    .st-key-floating_chat_panel,
    div:has(> #floating-chat-anchor) {
        position: fixed !important;
        top: 76px !important;
        left: 24px !important;
        width: 440px !important;
        max-width: min(440px, 92vw) !important;
        max-height: calc(100vh - 100px) !important;
        z-index: 100 !important;
        background: rgba(13, 17, 23, 0.86) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.13) !important;
        border-radius: 18px !important;
        padding: 20px 22px 24px 22px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.70), 0 0 24px rgba(58, 160, 255, 0.12) !important;
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

    /* User Query Display Card */
    .user-query-card {
        background: rgba(22, 30, 44, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-left: 3px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    .user-query-label {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #38bdf8 !important;
        margin-bottom: 4px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .user-query-text {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #F1F5F9 !important;
        line-height: 1.45 !important;
    }

    /* Supported Answer Card */
    .answer-card {
        background: rgba(15, 23, 36, 0.88) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-left: 3px solid #3AA0FF !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35) !important;
    }
    .answer-header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 10px !important;
        padding-bottom: 6px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    .answer-badge {
        font-size: 0.70rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        color: #38bdf8 !important;
        display: flex !important;
        align-items: center !important;
        gap: 5px !important;
    }
    .answer-verified {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        color: #34d399 !important;
        background: rgba(16, 185, 129, 0.12) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        padding: 2px 7px !important;
        border-radius: 12px !important;
    }
    .answer-body {
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
        color: #F8FAFC !important;
        letter-spacing: 0.012em !important;
    }
    .answer-lead {
        margin: 0 0 10px 0 !important;
        font-weight: 500 !important;
        color: #F8FAFC !important;
    }
    .answer-paragraph {
        margin: 0 0 10px 0 !important;
        color: #F8FAFC !important;
    }
    .answer-paragraph:last-child {
        margin-bottom: 0 !important;
    }
    .answer-list {
        list-style: none !important;
        padding: 0 !important;
        margin: 6px 0 10px 0 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    .answer-list li {
        display: flex !important;
        align-items: flex-start !important;
        gap: 10px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .list-badge {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        background: rgba(56, 189, 248, 0.18) !important;
        color: #38bdf8 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        margin-top: 2px !important;
        flex-shrink: 0 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
    }
    .list-text {
        color: #F8FAFC !important;
        font-size: 0.91rem !important;
        line-height: 1.55 !important;
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

    /* Citation Hierarchy Card */
    .citation-card {
        background: rgba(14, 20, 30, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 9px !important;
    }
    .citation-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 8px !important;
    }
    .citation-cell {
        background: rgba(22, 30, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 7px !important;
        padding: 7px 10px !important;
    }
    .citation-cell-full {
        grid-column: 1 / -1 !important;
    }
    .citation-label {
        display: block !important;
        font-size: 0.66rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #64748b !important;
        margin-bottom: 2px !important;
    }
    .citation-val-doc {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.80rem !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    .citation-val-badge {
        display: inline-block !important;
        background: rgba(56, 189, 248, 0.12) !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
        border-radius: 5px !important;
        padding: 2px 7px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
    .citation-val-section {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    .citation-excerpt-box {
        background: rgba(10, 14, 22, 0.75) !important;
        border-left: 3px solid #38bdf8 !important;
        border-radius: 6px !important;
        padding: 10px 12px !important;
    }
    .citation-excerpt-text {
        font-size: 0.82rem !important;
        line-height: 1.6 !important;
        color: #cbd5e1 !important;
        font-style: italic !important;
        margin: 0 !important;
        max-height: 120px !important;
        overflow-y: auto !important;
        word-break: break-word !important;
    }
    .citation-quote-glyph {
        color: #38bdf8 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        margin-right: 4px !important;
    }

    /* Expander Overrides inside Floating Chat Panel */
    .st-key-floating_chat_panel [data-testid="stExpander"],
    .st-key-floating_chat_panel .stExpander,
    .st-key-floating_chat_panel details {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        margin-bottom: 6px !important;
    }
    .st-key-floating_chat_panel [data-testid="stExpander"] summary,
    .st-key-floating_chat_panel .stExpander summary,
    .st-key-floating_chat_panel .streamlit-expanderHeader {
        background: rgba(20, 26, 38, 0.70) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 9px !important;
        padding: 6px 12px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        transition: all 0.2s ease !important;
    }
    .st-key-floating_chat_panel [data-testid="stExpander"] summary:hover,
    .st-key-floating_chat_panel .stExpander summary:hover {
        background: rgba(28, 38, 54, 0.85) !important;
        border-color: rgba(58, 160, 255, 0.35) !important;
        color: #ffffff !important;
    }
    .st-key-floating_chat_panel [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    .st-key-floating_chat_panel .stExpander [data-testid="stExpanderDetails"],
    .st-key-floating_chat_panel .streamlit-expanderContent {
        background: transparent !important;
        border: none !important;
        padding: 4px 0 8px 0 !important;
    }
    details:not([open]) > *:not(summary) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }

    /* Scrollable Conversation Container inside Panel */
    .st-key-chat_conversation_scroll {
        max-height: calc(100vh - 350px) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 4px !important;
        margin-bottom: 10px !important;
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
            top: auto !important;
            left: 12px !important;
            bottom: 12px !important;
            width: calc(100vw - 24px) !important;
            max-width: calc(100vw - 24px) !important;
            max-height: 65vh !important;
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

    # Active Conversation & Answer Display Area (Scrollable Container)
    with st.container(key="chat_conversation_scroll"):
        if st.session_state.active_response is not None:
            resp: RAGResponse = st.session_state.active_response

            # Show active query
            if st.session_state.active_query_text:
                st.markdown(
                    f"""
                    <div class="user-query-card">
                        <div class="user-query-label">💬 Engineering Query</div>
                        <div class="user-query-text">{html.escape(st.session_state.active_query_text)}</div>
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
                # Supported Answer with structured typography
                formatted_answer = format_answer_html(resp.answer)
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-header">
                            <span class="answer-badge">⚙️ GROUNDED ANSWER</span>
                            <span class="answer-verified">✓ GER-3620 Grounded</span>
                        </div>
                        <div class="answer-body">
                            {formatted_answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Collapsible Source Citation with clear 4-tier hierarchy
                printed_disp = (
                    f"Page {resp.printed_page}"
                    if resp.printed_page and resp.printed_page != "Unavailable"
                    else "N/A"
                )
                with st.expander("📌 Supporting Source Citation", expanded=True):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-grid">
                                <div class="citation-cell citation-cell-full">
                                    <span class="citation-label">DOCUMENT</span>
                                    <span class="citation-val-doc">{html.escape(str(resp.source))}</span>
                                </div>
                                <div class="citation-cell">
                                    <span class="citation-label">PHYSICAL PDF</span>
                                    <span class="citation-val-badge">Page {resp.page} of 60</span>
                                </div>
                                <div class="citation-cell">
                                    <span class="citation-label">PRINTED PAGE</span>
                                    <span class="citation-val-badge">{html.escape(printed_disp)}</span>
                                </div>
                            </div>
                            
                            <div class="citation-cell">
                                <span class="citation-label">SECTION HEADING</span>
                                <span class="citation-val-section">§ {html.escape(str(resp.section))}</span>
                            </div>
                            
                            <div class="citation-excerpt-box">
                                <span class="citation-label">VERBATIM SUPPORTING EXCERPT</span>
                                <blockquote class="citation-excerpt-text">
                                    <span class="citation-quote-glyph">“</span>{html.escape(str(resp.excerpt))}<span class="citation-quote-glyph">”</span>
                                </blockquote>
                            </div>
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
