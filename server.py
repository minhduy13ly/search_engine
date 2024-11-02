from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from reranker import ReRanker
from rawSearch import TFIDF

# Define FastAPI instance
app = FastAPI()

tf_idf = TFIDF()
re_ranker = ReRanker()


# Define the data model for requests
class QueryRequest(BaseModel):
    query: str

# Define the POST endpoint
@app.post("/api/search")
async def handle_query(query_request: QueryRequest):
    query = query_request.query

    if not query:
        return {"error": "No query provided"}, 400

    # Process the query
    clean_query = process_query(query)

    # Mock search and ranking (replace with your logic)
    filtered_results = tf_idf.search(clean_query, 5)
    doc_id, score, text = re_ranker.rank(query, filtered_results)

    # Return the result
    return {"doc_id": int(doc_id), "score": float(score), "text": text}

def process_query(query):
    # Basic query processing logic (expand as needed)
    return query

# Entry point to run the FastAPI app
if __name__ == "__main__":
    # Uvicorn runs the FastAPI app when you run `python serve.py`
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)