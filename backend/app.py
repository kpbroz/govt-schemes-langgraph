from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from graph.graph import graph


load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Input model for API
class Query(BaseModel):
    question: str



@app.on_event("startup")

def startup_event():
    print("Server has started!")
    
@app.get("/")
async def welcome():
    return {"greeting": "Hi bvc!"}
    

# Endpoint to process queries
@app.post("/ask/")
async def ask_question(query: Query):
    res = graph.invoke(input={"question": query.question})
    generation = res.get("generation", "No response generated.")
    return {"response": generation}

