from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


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


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore, web search..
Route the user question to either web search or vectorstore based on the following conditions.
The vectorstore contains documents related to Government of 3 schemes namely Pradhan Mantri Jan Arogya Yojana(Ayushman bharat), AMRIT BHARAT STATIONS (launched for development of Railway stations on Indian Railways) and One Nation One Subscription (Empowering India's Research Ecosystem). Use the vectorstore for questions on these topics. For all other schems, policies, bills, acts and campaigns of government of India, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
