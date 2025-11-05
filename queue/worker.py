import os, json, time, re
import pika, requests
from bs4 import BeautifulSoup
from dask import delayed, compute
from dotenv import load_dotenv
import psycopg2
from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
from pymongo import MongoClient, ReturnDocument
from urllib.parse import urlparse

load_dotenv()
RABBITMQ_URL = os.getenv("RABBITMQ_URL")

PG_CONN = dict(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    dbname=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
)
print("DEBUG: PG_PASSWORD =", PG_CONN["password"])  # TEMP

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = os.getenv("MONGO_DB")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36")
}

# --- DB clients ---
pg = psycopg2.connect(**PG_CONN)
pg.autocommit = True
mongo = MongoClient(MONGO_URI)[MONGO_DB]
clean_col = mongo["clean_pages"]
counters_col = mongo["counters"]

def store_raw_postgres(url: str, html: str):
    with pg.cursor() as cur:
        cur.execute("INSERT INTO raw_pages (url, html) VALUES (%s, %s)", (url, html))

def allocate_global_ids(n: int) -> int:
    """
    Atomically reserve n sequential global part numbers.
    Returns the starting index (0-based) before increment.
    """
    doc = counters_col.find_one_and_update(
        {"_id": "qa_seq"},
        {"$inc": {"seq": n}},
        upsert=True,
        return_document=ReturnDocument.BEFORE
    )
    start = (doc["seq"] if doc and "seq" in doc else 0)
    return start  # caller will add +1 when displaying as 1-based

def clean_and_make_qa(html: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "svg"]):
        tag.decompose()

    # Clean text
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))

    # Split into sentences or short chunks (keep it simple)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    sentences = sentences[:10]  # keep first 10 for demo

    # Allocate global IDs once for this page
    start_idx = allocate_global_ids(len(sentences))  # 0-based

    qa_pairs = []
    netloc = urlparse(url).netloc

    for i, s in enumerate(sentences):
        global_part = start_idx + i + 1  # 1-based across ALL pages
        local_part = i + 1               # 1-based within THIS page

        # IMPORTANT: Use the global part number in the question so each is unique
        q = f"What does the page say in part {global_part}?"
        a = s

        qa_pairs.append({
            "question": q,
            "answer": a,
            "meta": {
                "url": url,
                "domain": netloc,
                "local_part": local_part,
                "global_part": global_part
            }
        })

    return qa_pairs

@delayed
def scrape_and_store(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    html = r.text
    # 1) raw -> Postgres
    store_raw_postgres(url, html)
    # 2) cleaned -> Mongo
    qa_pairs = clean_and_make_qa(html, url)
    clean_col.insert_one({"url": url, "qa_pairs": qa_pairs, "ts": time.time()})
    return {"url": url, "qa_count": len(qa_pairs)}

# --- RabbitMQ consumer loop (batch for Dask) ---
def main():
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue="urls", durable=True)
    print("👂 waiting for messages on 'urls'... (Ctrl+C to stop)")

    try:
        while True:
            batch = []
            for _ in range(10):
                method, props, body = ch.basic_get(queue="urls", auto_ack=False)
                if not body:
                    break
                msg = json.loads(body.decode("utf-8"))
                url = msg["url"]
                batch.append((method.delivery_tag, url))
            if not batch:
                time.sleep(1)
                continue

            print(f"⚙️ processing batch of {len(batch)} urls with Dask...")
            tasks = [scrape_and_store(url) for _, url in batch]
            results = compute(*tasks)

            # ack after processing
            for (delivery_tag, _), res in zip(batch, results):
                print(" done:", res)
                ch.basic_ack(delivery_tag)
    finally:
        conn.close()

if __name__ == "__main__":
    
    main()
