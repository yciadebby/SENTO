import pytest

from ai.generation import GenerationError
from ai.pipeline import RAGPipeline
from ai.validator import EvidenceValidator, format_citations
from backend.chatbot_service import ChatbotService


def evidence(senator_id="risa-hontiveros", chunk_id="record-1", source_url="https://senate.gov.ph/record.pdf"):
    return {
        "chunk_id": chunk_id,
        "senator_id": senator_id,
        "senator_ids": [senator_id],
        "section": "Voting_Records",
        "text": "A reviewed journal records the senator's vote on this measure.",
        "verification_status": "verified",
        "citation_sources": [{
            "source_id": "senate-journal-2026-001",
            "title": "Senate Journal, 20th Congress",
            "issuing_body": "Senate of the Philippines",
            "source_type": "official_record",
            "source_url": source_url,
            "publication_date": "2026-08-01",
            "retrieved_at": "2026-11-25",
            "relevant_period": "20th Congress, First Regular Session",
            "locator": "p. 41, roll-call table",
            "verification_status": "verified",
        }],
    }


class FixedRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, senator_id, top_k=5, relevant_period=None):
        self.calls.append((query, senator_id, top_k, relevant_period))
        return self.documents


class CitingGenerator:
    def __init__(self, answer="The journal records a Yes vote. [E1]"):
        self.answer = answer
        self.calls = []

    def generate(self, query, senator, context, style):
        self.calls.append((query, senator, context, style))
        return self.answer


def test_selection_and_missing(senators):
    assert senators.get("risa-hontiveros")["minority_membership"] == "Unverified"
    assert len(senators.search("RISa")) == 1
    assert senators.search("nonexistent") == []
    with pytest.raises(ValueError):
        senators.get("not-a-senator")


def test_answer_uses_verified_retrieved_evidence_and_citations(senators):
    retriever = FixedRetriever([evidence()])
    generator = CitingGenerator()
    bot = ChatbotService(senators, RAGPipeline(retriever, generator, EvidenceValidator()))

    response = bot.ask("How did she vote on this measure?", "risa-hontiveros")

    assert response["answer"] == "The journal records a Yes vote. [E1]"
    assert response["label"] == "SOURCE-GROUNDED AI SUMMARY"
    assert response["validation"]["status"] == "CITATIONS_VALIDATED"
    assert response["validation"]["verified"] is False
    assert response["validation"]["semantic_support_checked"] is False
    assert response["citations"][0]["title"] == "Senate Journal, 20th Congress"
    assert response["citations"][0]["locator"] == "p. 41, roll-call table"
    assert response["citations"][0]["url"] == "https://senate.gov.ph/record.pdf"


def test_no_evidence_abstains_without_calling_generator(senators):
    retriever = FixedRetriever([])
    generator = CitingGenerator()
    bot = ChatbotService(senators, RAGPipeline(retriever, generator))

    response = bot.ask("What is the record?", "risa-hontiveros")

    assert "could not find a passage in the current project-curated notes" in response["answer"]
    assert response["validation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert response["citations"] == []
    assert generator.calls == []


def test_default_service_does_not_use_demo_records(senators):
    response = ChatbotService(senators).ask("bills", "risa-hontiveros")
    assert response["validation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert response["citations"] == []
    assert "does not establish" in response["answer"]


def test_injected_retriever_cannot_leak_another_senators_context(senators):
    retriever = FixedRetriever([evidence("francis-pangilinan")])
    generator = CitingGenerator()
    response = ChatbotService(senators, RAGPipeline(retriever, generator)).ask(
        "What is the record?", "risa-hontiveros"
    )
    assert response["retrieval_count"] == 0
    assert response["citations"] == []
    assert generator.calls == []


def test_uncited_or_unknown_marker_answer_is_withheld(senators):
    for answer in ("A factual answer with no citation.", "A claim cites an unknown source. [E2]"):
        response = ChatbotService(
            senators,
            RAGPipeline(FixedRetriever([evidence()]), CitingGenerator(answer)),
        ).ask("What is the record?", "risa-hontiveros")
        assert response["label"] == "ANSWER NOT VALIDATED"
        assert response["citations"] == []
        assert response["validation"]["status"] in {"UNCITED_ANSWER", "INVALID_CITATION"}


def test_invalid_source_url_is_removed_before_generation(senators):
    generator = CitingGenerator()
    response = ChatbotService(
        senators,
        RAGPipeline(FixedRetriever([evidence(source_url="javascript:alert(1)")]), generator),
    ).ask("What is the record?", "risa-hontiveros")
    assert response["validation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert response["citations"] == []
    assert generator.calls == []


def test_model_failure_returns_no_answer(senators):
    class BrokenGenerator:
        def generate(self, *args):
            raise GenerationError("private diagnostic")

    response = ChatbotService(
        senators,
        RAGPipeline(FixedRetriever([evidence()]), BrokenGenerator()),
    ).ask("What is the record?", "risa-hontiveros")
    assert response["validation"]["status"] == "SERVICE_UNAVAILABLE"
    assert response["citations"] == []
    assert "private diagnostic" not in response["answer"]


@pytest.mark.parametrize("query", ["", " ", "x" * 4001])
def test_invalid_question(query):
    with pytest.raises(ValueError):
        ChatbotService().ask(query, "risa-hontiveros")
