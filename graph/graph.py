from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import RELEVANCE, GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes.retrieve import retrieve
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.web_search import web_search
from graph.nodes.grade_relevance import grade_relevance
from graph.state import GraphState

load_dotenv()
# memory = SqliteSaver.from_conn_string(":memory:")
# memory = MemorySaver()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    question = state["question"]
    relevant = state["relevant"]
    
    if relevant == True:
        print("---QUESTION IS RELEVANT---")
        print("---ROUTE QUESTION---")
        source: RouteQuery = question_router.invoke({"question": question})
        if source.datasource == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return WEBSEARCH
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return RETRIEVE
    else:
        print("---QUESTION IS NOT RELEVANT---")
        return END



workflow = StateGraph(GraphState)
workflow.add_node(RELEVANCE, grade_relevance)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)


workflow.set_entry_point(RELEVANCE)

workflow.add_conditional_edges(RELEVANCE, 
                               route_question,
                               path_map={
                                WEBSEARCH: WEBSEARCH,
                                RETRIEVE: RETRIEVE,
                                END: END,
                               })


workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)


# app = workflow.compile(checkpointer=memory)
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")