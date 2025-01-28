from typing import Dict, Any

from graph.state import GraphState
from graph.chains.relevance_grader import relevance_grader


def grade_relevance(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the question is releted to the schems, policies, bills, acts and campaigns of government of India
    If any document is not relavant, we will set a flag to run web search

    Args:
        state (dict): The current state graph

    Returns:
        state (dict): updated generation and relevant state
    """

    question = state["question"]

    score = relevance_grader.invoke({"question": question})

    grade = score.binary_score

    if grade == True:
        print("---GRADE: QUESTION RELEVANT---")
        return {"question": question, "relevant": True}
    else:
        print("---GRADE: QUESTION NOT RELEVANT---")
        return {
            "question": question,
            "relevant": False,
            "generation": "Question is not related to schems, policies, bills, acts and campaigns of Government of India. Please ask relevant questions.",
        }
