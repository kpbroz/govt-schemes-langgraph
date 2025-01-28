from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from langchain_openai import AzureChatOpenAI


OPENAI_MODEL = os.getenv("OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_AZURE_ENDPOINT = os.getenv("OPENAI_AZURE_ENDPOINT")

llm = AzureChatOpenAI(
    temperature=0,
    stop=["\nObservation"],
    model=OPENAI_MODEL,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_AZURE_ENDPOINT,
)


class GradeDocuments(BaseModel):
    """Binary score for relavance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relavent to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """ You are a grader assessing the relavance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relavent. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relavent to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
