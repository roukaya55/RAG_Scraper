import os, json
import pika
from dotenv import load_dotenv

load_dotenv()
RABBITMQ_URL = os.getenv("RABBITMQ_URL")

urls = [
    "https://techcrunch.com/2025/10/28/openai-completes-its-for-profit-recapitalization/",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Meta"
]

params = pika.URLParameters(RABBITMQ_URL)
conn = pika.BlockingConnection(params)
ch = conn.channel()
ch.queue_declare(queue="urls", durable=True)

for u in urls:
    ch.basic_publish(
        exchange="",
        routing_key="urls",
        body=json.dumps({"url": u}),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print(f" queued: {u}")

conn.close()
