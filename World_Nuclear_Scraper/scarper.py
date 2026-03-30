import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://world-nuclear.org"
LISTING_URL = "https://world-nuclear.org/news-and-media?pageNumber={}"

OUTPUT_DIR = "articles"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# -----------------------------
# CLEAN FILENAME
# -----------------------------
def clean_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name[:150]


# -----------------------------
# GET ARTICLE LINKS FROM PAGE
# -----------------------------
def get_article_links(page_num):
    url = LISTING_URL.format(page_num)
    print(f"Scraping page {page_num}...")

    res = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/news-and-media/" in href and href != "/news-and-media":
            full_url = urljoin(BASE_URL, href)

            # avoid pagination links
            if "pageNumber=" in full_url:
                continue

            links.add(full_url)

    return list(links)


# -----------------------------
# EXTRACT ONLY ARTICLE TEXT
# -----------------------------
def get_article_text(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        container = soup.select_one("#country_content_wrapper .country_content")

        if not container:
            return ""

        paragraphs = container.find_all("p")

        clean_paragraphs = []

        for p in paragraphs:
            text = p.get_text(strip=True)

            if not text:
                continue

            # remove junk
            if len(text) < 40:
                continue

            if any(x in text.lower() for x in [
                "world nuclear news",
                "contact",
                "follow us",
                "related",
                "read more",
                "share this",
                "copyright"
            ]):
                continue

            clean_paragraphs.append(text)

        return "\n\n".join(clean_paragraphs)

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


# -----------------------------
# MAIN SCRAPER
# -----------------------------
def scrape_all(max_pages=100):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seen_links = set()
    article_count = 0

    for page in range(1, max_pages + 1):
        links = get_article_links(page)

        if not links:
            print("No more pages.")
            break

        for link in links:
            if link in seen_links:
                continue

            seen_links.add(link)

            text = get_article_text(link)

            if not text:
                continue

            filename = clean_filename(link.split("/")[-1]) + ".txt"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            article_count += 1
            print(f"Saved {article_count}: {filename}")

    print(f"\nDone. Scraped {article_count} articles.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    scrape_all(max_pages=200)
