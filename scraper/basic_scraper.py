import requests
from bs4 import BeautifulSoup

def scrape_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # throws an error if request fails

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for s in soup(["script", "style"]):
        s.decompose()

    text = " ".join(soup.get_text().split())
    return text


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/OpenAI"
    content = scrape_page(url)
    print(content[:1000])  # print first 1000 characters only
