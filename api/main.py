# api.py / main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
from datetime import datetime

# ✅ Import RAG engine (already loads DB + embeddings)
from rag.rag_engine import smart_retrieval, qa_dict, vectorstore

# ---------------------- FastAPI Setup ----------------------
app = FastAPI(
    title="RAG Scraper API",
    description="API for querying scraped and processed content",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- API Key Auth ----------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
VALID_API_KEYS = {"student-key-123": "Roukaya", "teacher-key-456": "Instructor"}

request_counts = {}

def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid API Key")

def rate_limit(api_key: str, limit: int = 50, window: int = 3600):
    now = time.time()
    key = f"{api_key}_{int(now // window)}"
    request_counts[key] = request_counts.get(key, 0) + 1
    if request_counts[key] > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# ---------------------- Models ----------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


# ---------------------- Routes ----------------------
@app.get("/")
def home():
    return {"message": "RAG Scraper API working 🎯"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_pairs": len(qa_dict),
        "vector_store_ready": True,
        "time": datetime.utcnow()
    }
@app.post("/query")
def query(data: QueryRequest, api_key: str = Depends(get_api_key)):
    rate_limit(api_key)

    results = smart_retrieval(data.question, data.top_k)

    formatted = [
        {
            "answer": r["answer"],
            "question": r.get("question"),
            "url": r.get("meta", {}).get("url") if "meta" in r else r.get("url"),
            "global_part": r.get("meta", {}).get("global_part") if "meta" in r else r.get("global_part"),
            "exact_match": r.get("exact")
        }
        for r in results
    ]

    return {
        "query": data.question,
        "results": formatted,
        "count": len(results),
        "timestamp": datetime.utcnow()
    }


@app.post("/search")
def search(data: SearchRequest, api_key: str = Depends(get_api_key)):
    rate_limit(api_key)

    matches = [
        q for q, (a, m) in qa_dict.items()
        if data.query.lower() in q.lower() or data.query.lower() in a.lower()
    ][:data.limit]

    return {"query": data.query, "results": matches}

@app.get("/raw-data")
def raw(limit: int = 5, api_key: str = Depends(get_api_key)):
    rate_limit(api_key)

    data = list(qa_dict.items())[:limit]
    return {"count": len(data), "data": data}
