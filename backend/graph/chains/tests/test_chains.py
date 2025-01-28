from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.chains.relevance_grader import GradeRelevance, relevance_grader


def test_retrieval_grader_answer_yes() -> None:
    question = "what is ayushman bharat"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "what is ayushman bharat"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "how to get scheme benifits"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "what are the schmes of government of india"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "what is the use of AMRIT BHARAT STATIONS"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "how to empower india research ecosystem"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "what is swaccha bharat"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
    
def test_relevance_grader_yes() -> None:
    question = "waqf bill"
    
    res: GradeRelevance = relevance_grader.invoke({"question": question})
    
    assert res.binary_score == True
    
    
def test_relevance_grader_no() -> None:
    question = "can you explain about nazi philosophy?"
    
    res: GradeRelevance = relevance_grader.invoke({"question": question})
    
    assert res.binary_score == False
    
