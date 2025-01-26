from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_AZURE_ENDPOINT = os.getenv("OPENAI_AZURE_ENDPOINT")

llm = AzureChatOpenAI(temperature=0, stop=["\nObservation"], model=OPENAI_MODEL, api_key=AZURE_OPENAI_API_KEY, api_version=OPENAI_API_VERSION, azure_endpoint=OPENAI_AZURE_ENDPOINT)


class GradeRelevance(BaseModel):
    "Binary score for whether question is releted to the policies, schemes and campaigns of government of India"
    
    binary_score: bool = Field(
        description="Question is eleted to the policies, schemes and campaigns of government of India or not, 'yes' or 'no'"
    )
    
structured_llm_grader = llm.with_structured_output(GradeRelevance)
    
system = """You are a grader assessing whether query or question asked by the user is related to the  policies, schemes and campaigns of government of India. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the question asked by the user is related to the  policies, schemes and campaigns of government of India."""

relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: \n\n {question}"),
    ]
)

relevance_grader: RunnableSequence = relevance_prompt | structured_llm_grader