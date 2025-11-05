# rag_engine.py
import os, re
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------
# Initialize RAG Engine ONCE
# ----------------------------

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "rag_scraper")

print("🔌 Connecting to MongoDB...")
mongo = MongoClient(MONGO_URI)[MONGO_DB]
clean_col = mongo["clean_pages"]

texts = []       # what we embed (answers only)
metadatas = []   # metadata for each QA pair
qa_dict = {}     # exact question lookup

count = 0
print("📥 Loading QA pairs from MongoDB...")
for doc in clean_col.find({}, {"qa_pairs": 1}):
    for p in doc["qa_pairs"]:
        q = (p.get("question") or "").strip()
        a = (p.get("answer") or "").strip()
        meta = p.get("meta") or {}

        if q and a:
            texts.append(a)
            metadatas.append({
                "question": q,
                "url": meta.get("url"),
                "domain": meta.get("domain"),
                "global_part": meta.get("global_part"),
                "local_part": meta.get("local_part"),
            })

            qa_dict[q.lower()] = (a, metadatas[-1])
            count += 1

print(f"✅ Loaded {count} QA pairs into memory")

print("🧠 Creating embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🗂 Building Chroma vector index in RAM...")
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

print("🚀 RAG Engine ready!")


# ----------------------------
# Helper functions
# ----------------------------

def extract_part_number(query: str):
    m = re.search(r"part\s+(\d+)", query.lower())
    return int(m.group(1)) if m else None


def smart_retrieval(query: str, k: int = 8):
    """Return best answer(s) from your QA dataset"""
    q_lower = query.lower()

    # ✅ Exact DB question match
    if q_lower in qa_dict:
        answer, meta = qa_dict[q_lower]
        return [{
            "answer": answer,
            "question": meta.get("question"),
            "url": meta.get("url"),
            "global_part": meta.get("global_part"),
            "exact_match": True
        }]

    # ✅ Vector search
    docs = vectorstore.similarity_search(query, k=k)
    target_part = extract_part_number(query)

    # ✅ If user asked 'part N', filter exact part
    if target_part is not None:
        filtered = [d for d in docs if d.metadata.get("global_part") == target_part]
        if filtered:
            docs = filtered

    # ✅ Return formatted response
    results = []
    for d in docs:
        results.append({
            "answer": d.page_content.strip(),
            "question": d.metadata.get("question"),
            "url": d.metadata.get("url"),
            "global_part": d.metadata.get("global_part"),
            "exact_match": False
        })
    return results


# ----------------------------
# Local test ability (optional)
# ----------------------------
if __name__ == "__main__":
    query = "What does the page say in part 8?"
    results = smart_retrieval(query)

    print(f"\n🧠 QUESTION: {query}\n")
    for r in results[:3]:
        print("✅", r["answer"])
        print("↪ URL:", r["url"])
        print("Part:", r["global_part"])
        print()
