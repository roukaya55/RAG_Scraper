from dask import delayed, compute
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@delayed
def scrape_page(url):
    response = requests.get(url, headers=HEADERS)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    return {"url": url, "content": " ".join(soup.get_text().split())}

urls = [
    "https://techcrunch.com/2025/10/28/openai-completes-its-for-profit-recapitalization/",
    "https://en.wikipedia.org/wiki/OpenAI",
]

tasks = [scrape_page(u) for u in urls]
results = compute(*tasks)
print(results)
