from typing import Dict, Any
from langchain.schema import Document

from graph.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults


web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] if "documents" in state else None
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"documents": documents, "question": question}
    
    