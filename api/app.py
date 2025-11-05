from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import time
from datetime import datetime

# Import your RAG components
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(
    title="RAG Scraper API",
    description="API for querying scraped and processed content",
    version="1.0.0"
)

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


VALID_API_KEYS = {
    "student-key-123": "Roukaya",
    "teacher-key-456": "Instructor"
}

# ✅ Rate limiting storage
request_counts = {}

def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Validate API key from the request header.
    """
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
        headers={"WWW-Authenticate": "API key"},
    )

def rate_limit(api_key: str, limit: int = 100, window: int = 3600):
    """Simple rate limiting"""
    current_time = time.time()
    key = f"{api_key}_{current_time // window}"
    
    if key not in request_counts:
        request_counts[key] = 0
    
    request_counts[key] += 1
    
    if request_counts[key] > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# ✅ Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    relevant_documents: List[str]
    document_count: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    total_documents: int
    vector_store_ready: bool
    timestamp: str

class DocumentResponse(BaseModel):
    id: int
    content: str
    source: str

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

# ✅ Initialize RAG components
print("Initializing RAG system...")

with open("new_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [f"Q: {p['question']}\nA: {p['answer']}" for item in data for p in item["qa_pairs"]]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text("\n\n".join(documents))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(chunks, embeddings)
print("RAG system initialized successfully!")

# ✅ RAG function (your existing logic)
def smart_retrieval_answer(question: str, k: int = 3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    if not relevant_docs:
        return "No relevant information found.", []
    answers = []
    for doc in relevant_docs:
        if "A: " in doc.page_content:
            answer_part = doc.page_content.split("A: ")[1]
            answers.append(answer_part.split("\n")[0])
    if answers:
        return answers[0], relevant_docs
    return relevant_docs[0].page_content[:200], relevant_docs

# ✅ API Endpoints (meeting all requirements)

@app.get("/", response_model=dict)
async def root():
    return {"message": "RAG Scraper API is running!"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - REQUIRED"""
    return HealthResponse(
        status="healthy",
        total_documents=len(documents),
        vector_store_ready=True,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/raw-data", response_model=List[DocumentResponse])
async def get_raw_data(
    limit: int = 10,
    offset: int = 0,
    api_key: str = Depends(get_api_key)
):
    """✅ Fetching raw scraped data - REQUIRED"""
    rate_limit(api_key)
    
    end_index = min(offset + limit, len(documents))
    response_docs = []
    
    for i, doc in enumerate(documents[offset:end_index], start=offset):
        response_docs.append(
            DocumentResponse(
                id=i,
                content=doc,
                source="scraped_qa_dataset"
            )
        )
    
    return response_docs

@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest, 
    api_key: str = Depends(get_api_key)
):
    """✅ Querying processed/enhanced content - REQUIRED"""
    rate_limit(api_key)
    
    try:
        answer, retrieved_docs = smart_retrieval_answer(
            request.question, 
            k=request.top_k
        )
        
        doc_contents = [doc.page_content for doc in retrieved_docs]
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            relevant_documents=doc_contents,
            document_count=len(retrieved_docs),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/search", response_model=List[DocumentResponse])
async def search_documents(
    request: SearchRequest,
    api_key: str = Depends(get_api_key)
):
    """✅ Searching indexed data - REQUIRED"""
    rate_limit(api_key)
    
    matching_docs = []
    for i, doc in enumerate(documents):
        if request.query.lower() in doc.lower():
            matching_docs.append(
                DocumentResponse(
                    id=i,
                    content=doc,
                    source="scraped_qa_dataset"
                )
            )
        if len(matching_docs) >= request.limit:
            break
    
    return matching_docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)