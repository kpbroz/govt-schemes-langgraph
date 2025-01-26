from typing import Dict, Any

from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relavent to the question
    If any document is not relavent, we will set a flag to run web search

    Args:
        state (dict): The current state graph

    Returns:
        state (dict): Filtered out irrelavent documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELAVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELAVENT---")
            filtered_docs.append(d)

        else:
            print("---GRADE: DOCUMENT NOT RELAVENT---")
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
