from dask import delayed, compute
import requests
from bs4 import BeautifulSoup
import json
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

@delayed
def scrape_page(url):
    response = requests.get(url, headers=HEADERS)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Remove useless tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "svg"]):
        tag.decompose()

    # Get clean text
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))

    # Split into smaller logical chunks (for QA)
    sentences = text.split(". ")
    qa_pairs = []

    for i, s in enumerate(sentences[:10]):  # just take first few for demo
        question = f"What does the page say in part {i+1}?"
        answer = s.strip()
        qa_pairs.append({"question": question, "answer": answer})

    return {"url": url, "qa_pairs": qa_pairs}

urls = [
    "https://techcrunch.com/2025/10/28/openai-completes-its-for-profit-recapitalization/",
    "https://en.wikipedia.org/wiki/OpenAI",
]

tasks = [scrape_page(u) for u in urls]
results = compute(*tasks)

# Save to JSON
with open("qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Q/A dataset saved to qa_dataset.json")
