import os
import re
import time
import hashlib
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests

BASE_URL = "https://www.ans.org"
OUTPUT_FOLDER = "ans_articles"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TARGET_COUNT = 50
visited_urls = set()
content_hashes = set()

HEADERS = {"User-Agent": "Mozilla/5.0"}

SENTIMENT_KEYWORDS = [
    "public opinion", "public perception", "public sentiment",
    "support for nuclear", "anti-nuclear", "pro-nuclear",
    "public trust", "risk perception", "nuclear energy debate",
    "energy controversy", "community response", "public acceptance",
    "industry reputation", "advocacy",
]

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def content_matches(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SENTIMENT_KEYWORDS)

def hash_content(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def scrape_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        title_tag = soup.find("h1")
        if not title_tag:
            return False
        title = clean_filename(title_tag.get_text())

        article_tag = soup.find("article", class_="article")
        if not article_tag:
            return False

        text_container = article_tag.find("div", class_="text")
        if not text_container:
            return False

        all_paragraphs = []

        # Collect paragraphs from both possible paths
        for cls in ["initial", "expand"]:
            path = text_container.find("div", class_=cls)
            if path:
                copy_div = path
                for sub_cls in ["content", "page-box", "copy"]:
                    next_div = copy_div.find("div", class_=sub_cls)
                    if next_div:
                        copy_div = next_div
                paragraphs = copy_div.find_all("p")
                all_paragraphs.extend([p.get_text().strip() for p in paragraphs])

        if not all_paragraphs:
            return False

        content = "\n".join(all_paragraphs)
        if len(content) < 200:
            return False

        # Optional: filter by keywords
        if not content_matches(content):
            return False

        # Deduplication
        content_hash = hash_content(content)
        if content_hash in content_hashes:
            return False
        content_hashes.add(content_hash)

        file_path = os.path.join(OUTPUT_FOLDER, f"{title}.txt")
        if os.path.exists(file_path):
            return False

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(title + "\n\n")
            f.write(content)

        print(f"Saved: {title}")
        return True

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return False

def crawl_news_with_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)
    driver.get(f"{BASE_URL}/news/")

    total_saved = 0

    while total_saved < TARGET_COUNT:
        # Wait for article elements to load
        wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "main.body div.page-box div.articles.list article.article")
        ))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.select("main.body div.page-box div.articles.list article.article")

        for article in articles:
            h2 = article.find("h2", class_="headline")
            if not h2:
                continue
            link_tag = h2.find("a")
            if not link_tag or not link_tag.get("href"):
                continue
            link = urljoin(BASE_URL, link_tag["href"])
            if link in visited_urls:
                continue
            visited_urls.add(link)

            if scrape_article(link):
                total_saved += 1
            if total_saved >= TARGET_COUNT:
                break

        # Click "More Articles" button if exists
        try:
            more_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-pill.blue")))
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(1)  # small pause to allow JS to load articles
        except:
            print("No more 'More Articles' button found.")
            break

    driver.quit()
    print(f"Finished. Total saved: {total_saved}")

if __name__ == "__main__":
    crawl_news_with_selenium()
