"""Automated test suite for Gas Turbine Maintenance Assistant RAG application.

Tests:
1. PDF extraction and page metadata preservation.
2. Document chunk creation.
3. ChromaDB indexing, retrieval, and cosine distance scores.
4. Persistence and reuse of an existing index (without re-embedding).
5. Answer generation and strict evidence validation.
6. Correct source citation formatting (PDF page, printed page, section).
7. Answerable questions return supported answers with citations.
8. Unanswerable questions return exactly 'Not found'.
9. Retrieved but irrelevant passages do not trigger fabricated answers.
10. Missing or invalid PDF produces clear, useful errors.
11. Verification of all five required exam questions.
"""

from pathlib import Path
import pytest
from config import DEFAULT_PDF_PATH
from rag_pipeline import (
    RetrievedChunk,
    answer_question,
    chunk_documents,
    compute_file_hash,
    extract_pdf_documents,
    init_or_load_vectorstore,
    is_valid_pdf,
    retrieve_relevant_chunks,
    validate_evidence,
)


@pytest.fixture(scope="session")
def vectorstore_fixture():
    """Shared vectorstore fixture loaded from the official PDF."""
    assert is_valid_pdf(DEFAULT_PDF_PATH), f"PDF missing at {DEFAULT_PDF_PATH}"
    vs, stats = init_or_load_vectorstore(DEFAULT_PDF_PATH, force_reindex=False)
    return vs, stats


def test_pdf_validity_and_extraction():
    """Verify PDF extraction, page count, and metadata preservation."""
    assert is_valid_pdf(DEFAULT_PDF_PATH)
    docs = extract_pdf_documents(DEFAULT_PDF_PATH)
    assert len(docs) >= 50, f"Expected >= 50 pages, got {len(docs)}"

    # Check page 5 (Introduction)
    page_5 = next((d for d in docs if d.metadata.get("page") == 5), None)
    assert page_5 is not None
    assert page_5.metadata.get("printed_page") == "1"
    assert page_5.metadata.get("section") == "Introduction"
    assert "Maintenance costs and machine availability" in page_5.page_content


def test_chunk_creation():
    """Verify chunk creation preserves metadata and conforms to size limits."""
    docs = extract_pdf_documents(DEFAULT_PDF_PATH)
    chunks = chunk_documents(docs[:5], chunk_size=800, chunk_overlap=120)
    assert len(chunks) > len(docs[:5])

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "printed_page" in chunk.metadata
        assert "section" in chunk.metadata
        assert len(chunk.page_content) <= 1200  # Allow slight flexibility for boundary words


def test_persistence_and_reuse(tmp_path):
    """Verify that an existing ChromaDB index is persisted and reused without re-indexing."""
    # First build
    vs1, stats1 = init_or_load_vectorstore(
        DEFAULT_PDF_PATH, persist_dir=tmp_path / "test_chroma", force_reindex=True
    )
    assert stats1["reused"] is False
    assert stats1["chunk_count"] > 0

    # Second build (should reuse cached metadata and collection)
    vs2, stats2 = init_or_load_vectorstore(
        DEFAULT_PDF_PATH, persist_dir=tmp_path / "test_chroma", force_reindex=False
    )
    assert stats2["reused"] is True
    assert stats2["chunk_count"] == stats1["chunk_count"]
    assert stats2["file_hash"] == stats1["file_hash"]


def test_missing_or_invalid_pdf_error(tmp_path):
    """Verify that a missing or invalid PDF raises a useful FileNotFoundError."""
    non_existent = tmp_path / "non_existent.pdf"
    with pytest.raises(FileNotFoundError):
        extract_pdf_documents(non_existent)

    # Corrupt fake PDF
    fake_pdf = tmp_path / "corrupt.pdf"
    fake_pdf.write_text("This is an HTML 404 error page, not a real PDF.")
    with pytest.raises(FileNotFoundError):
        extract_pdf_documents(fake_pdf)


def test_retrieval_and_similarity(vectorstore_fixture):
    """Verify vector similarity retrieval returns relevant passages with valid distance scores."""
    vs, _ = vectorstore_fixture
    query = "Combustion inspection interval hours"
    retrieved = retrieve_relevant_chunks(vs, query, top_k=4)

    assert len(retrieved) == 4
    for r in retrieved:
        assert isinstance(r, RetrievedChunk)
        assert r.score >= 0.0  # Cosine distance
        assert r.source == DEFAULT_PDF_PATH.name
        assert r.page > 0


def test_question_1_three_inspection_types(vectorstore_fixture):
    """Q1: What are the three main types of inspection? Must return Combustion, Hot Gas Path, Major."""
    vs, _ = vectorstore_fixture
    res = answer_question("What are the three main types of inspection?", vs)

    assert res.is_supported is True
    assert res.answer != "Not found"
    assert "Combustion Inspection" in res.answer
    assert "Hot Gas Path Inspection" in res.answer
    assert "Major Inspection" in res.answer
    assert res.source == DEFAULT_PDF_PATH.name
    assert res.page in (25, 32, 34)  # Documented sections in GER-3620
    assert res.excerpt is not None


def test_question_2_maintenance_factor_of_two(vectorstore_fixture):
    """Q2: What does a maintenance factor of two mean? Must explain baseline and factor of 2 (interval halved)."""
    vs, _ = vectorstore_fixture
    res = answer_question("What does a maintenance factor of two mean?", vs)

    assert res.is_supported is True
    assert res.answer != "Not found"
    assert "maintenance factor" in res.answer.lower()
    assert any(term in res.answer.lower() for term in ["twice", "half", "divided by 2", "accumulate at twice"])
    assert res.source == DEFAULT_PDF_PATH.name
    assert res.page in (36, 40, 9)
    assert res.excerpt is not None


def test_question_3_purchase_price_not_found(vectorstore_fixture):
    """Q3: What is the purchase price of this gas turbine? Must return exactly 'Not found'."""
    vs, _ = vectorstore_fixture
    res = answer_question("What is the purchase price of this gas turbine?", vs)

    assert res.is_supported is False
    assert res.answer == "Not found"
    assert res.source is None
    assert res.excerpt is None


def test_question_4_titanium_melting_point_not_found(vectorstore_fixture):
    """Q4: What is the melting point of titanium? Must return exactly 'Not found'."""
    vs, _ = vectorstore_fixture
    res = answer_question("What is the melting point of titanium?", vs)

    assert res.is_supported is False
    assert res.answer == "Not found"
    assert res.source is None
    assert res.excerpt is None


def test_question_5_combustion_inspection_interval(vectorstore_fixture):
    """Q5: What is the recommended combustion-inspection interval in hours? Must distinguish model-specific guidance."""
    vs, _ = vectorstore_fixture
    res = answer_question("What is the recommended combustion-inspection interval in hours?", vs)

    assert res.is_supported is True
    assert res.answer != "Not found"
    assert any(model in res.answer for model in ["6B", "7E", "9E", "7F", "DLN"])
    assert any(hrs in res.answer for hrs in ["8,000", "12,000", "24,000", "32,000"])
    assert res.source == DEFAULT_PDF_PATH.name
    assert res.page in (33, 36)
    assert res.excerpt is not None


def test_unsupported_out_of_scope_query(vectorstore_fixture):
    """Verify that an arbitrary out-of-scope question returns exactly 'Not found'."""
    vs, _ = vectorstore_fixture
    res = answer_question("What is the recipe for baking chocolate chip cookies?", vs)

    assert res.is_supported is False
    assert res.answer == "Not found"
    assert res.source is None
    assert res.excerpt is None


def test_empty_query(vectorstore_fixture):
    """Verify that an empty query returns 'Not found' safely."""
    vs, _ = vectorstore_fixture
    res = answer_question("   ", vs)

    assert res.is_supported is False
    assert res.answer == "Not found"


def test_gemini_integration_mock_grounded_answer(vectorstore_fixture):
    """Verify that Gemini generates answers with citations when evidence is supported."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    vs, _ = vectorstore_fixture
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="The three primary inspection types are Combustion, Hot Gas Path, and Major inspections."
    )

    res = answer_question(
        "What are the three main types of inspection?",
        vs,
        gemini_llm=mock_llm,
        use_gemini=True,
    )

    assert res.is_supported is True
    assert "Combustion, Hot Gas Path, and Major" in res.answer
    assert res.source == DEFAULT_PDF_PATH.name
    assert res.page is not None
    assert res.excerpt is not None
    assert mock_llm.invoke.called


def test_gemini_never_called_on_unsupported_evidence(vectorstore_fixture):
    """Critical: verify Gemini is NEVER invoked if evidence validation fails."""
    from unittest.mock import MagicMock

    vs, _ = vectorstore_fixture
    mock_llm = MagicMock()

    # Question 3: Purchase price (unsupported)
    res_q3 = answer_question(
        "What is the purchase price of this gas turbine?",
        vs,
        gemini_llm=mock_llm,
        use_gemini=True,
    )
    assert res_q3.is_supported is False
    assert res_q3.answer == "Not found"
    assert not mock_llm.invoke.called, "Gemini must not be called when evidence check fails!"

    # Question 4: Titanium melting point (unsupported)
    res_q4 = answer_question(
        "What is the melting point of titanium?",
        vs,
        gemini_llm=mock_llm,
        use_gemini=True,
    )
    assert res_q4.is_supported is False
    assert res_q4.answer == "Not found"
    assert not mock_llm.invoke.called, "Gemini must not be called when evidence check fails!"


def test_gemini_not_found_handling(vectorstore_fixture):
    """Verify that if Gemini returns 'Not found', response is marked unsupported."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    vs, _ = vectorstore_fixture
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Not found")

    res = answer_question(
        "What are the three main types of inspection?",
        vs,
        gemini_llm=mock_llm,
        use_gemini=True,
    )

    assert res.is_supported is False
    assert res.answer == "Not found"


def test_gemini_fallback_to_extractive_on_exception(vectorstore_fixture):
    """Verify clean fallback to local extractive answerer if Gemini API fails."""
    from unittest.mock import MagicMock

    vs, _ = vectorstore_fixture
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Gemini API connection error")

    res = answer_question(
        "What are the three main types of inspection?",
        vs,
        gemini_llm=mock_llm,
        use_gemini=True,
    )

    # Should fall back to extractive generator seamlessly
    assert res.is_supported is True
    assert "Combustion Inspection" in res.answer
    assert res.source == DEFAULT_PDF_PATH.name
    assert res.page is not None
