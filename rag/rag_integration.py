from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import json

# 1️⃣ Load your data
with open("new_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
for item in data:
    for pair in item["qa_pairs"]:
        text = f"Q: {pair['question']}\nA: {pair['answer']}"
        documents.append(text)

print(f"Loaded {len(documents)} QA pairs")

# 2️⃣ Split texts
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text("\n\n".join(documents))

# 3️⃣ FREE embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(chunks, embeddings)

# 4️⃣ Simple context-based answer (no LLM needed)
def smart_retrieval_answer(question, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    
    if not relevant_docs:
        return "No relevant information found.", []
    
    # Extract answers from the retrieved QA pairs
    answers = []
    for doc in relevant_docs:
        content = doc.page_content
        # Look for "A: " pattern in the content
        if "A: " in content:
            answer_part = content.split("A: ")[1]
            answers.append(answer_part.split("\n")[0])  # Get first line of answer
    
    if answers:
        # Return the most common or first answer
        return f"Based on the retrieved information: {answers[0]}", relevant_docs
    else:
        return f"Found relevant context: {relevant_docs[0].page_content[:200]}...", relevant_docs

# 5️⃣ Test
query = "What did the TechCrunch article say about OpenAI's new structure?"
answer, retrieved_docs = smart_retrieval_answer(query, vectorstore)

print("\n🔍 QUESTION:", query)
print("💬 ANSWER:", answer)
print(f"\n📄 Retrieved {len(retrieved_docs)} relevant documents")