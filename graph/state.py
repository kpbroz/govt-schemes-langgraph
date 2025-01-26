from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        relevant: question relevancy
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    relevant: bool
    generation: str
    web_search: bool
    documents: List[str]