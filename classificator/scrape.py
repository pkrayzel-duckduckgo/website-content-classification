import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import trafilatura
from transformers import pipeline
import pandas as pd
import csv


def fetch_and_save_html(url, output_file):
    """Fetch webpage content using a simple HTTP request."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"Plain HTML content saved to {output_file}")
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")


def render_and_save_html(url, output_file):
    """Render webpage using Selenium and save content with retries."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    retries = 3

    for attempt in range(retries):
        driver = webdriver.Chrome(service=service, options=options)
        try:
            driver.set_page_load_timeout(180)
            driver.get(url)
            time.sleep(5)

            html_content = driver.page_source
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(html_content)

            print(f"Rendered HTML content saved to {output_file}")
            return

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(10)
        finally:
            driver.quit()

    print(f"Failed to render {url} after {retries} attempts.")


def extract_title_from_html(file_path):
    """Extract the title from an HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')

            if soup.title and soup.title.string:
                return soup.title.string.strip()

            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()

            twitter_title = soup.find('meta', property='twitter:title')
            if twitter_title and twitter_title.get('content'):
                return twitter_title['content'].strip()

            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.get_text():
                return h1_tag.get_text().strip()

            return "Unknown Title"

    except Exception as e:
        print(f"Failed to extract title from {file_path}: {e}")
        return "Unknown Title"


def extract_sentences(html_content):
    """Extract sentences from HTML content."""
    text = trafilatura.extract(html_content)
    sentences = text.split('. ') if text else []
    return sentences


def classify_sentences(sentences, title, url, classifier):
    """Classify sentences as relevant or not."""
    results = []
    for sentence in sentences:
        try:
            context = f"Title: {title} URL: {url} Sentence: {sentence}"
            result = classifier(context)
            results.append((url, title, sentence, result[0]['label'], result[0]['score']))
        except Exception as e:
            print(f"Failed to classify sentence: {e}")
    return results


def process_urls(urls, output_dir="website_outputs", model_path="../preparation/fine_tuned_distilbert"):
    """Process multiple URLs by fetching, rendering, and classifying content."""
    os.makedirs(output_dir, exist_ok=True)
    classifier = pipeline("text-classification", model=model_path)

    summary_file = os.path.join(output_dir, "classification_summary.csv")
    detail_file = os.path.join(output_dir, "classification_details.csv")

    # Initialize CSV files with headers
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["url", "plain_relevant", "rendered_relevant", "difference"])
        writer.writeheader()

    with open(detail_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "sentence", "relevant", "score"])
        writer.writeheader()

    for url in urls:
        print(f"Processing {url}")

        safe_url = url.replace("https://", "").replace("http://", "").replace("/", "_")
        plain_file = os.path.join(output_dir, f"{safe_url}_plain.html")
        rendered_file = os.path.join(output_dir, f"{safe_url}_rendered.html")

        fetch_and_save_html(url, plain_file)
        render_and_save_html(url, rendered_file)

        with open(plain_file, "r", encoding="utf-8") as f:
            plain_sentences = extract_sentences(f.read())
        with open(rendered_file, "r", encoding="utf-8") as f:
            rendered_sentences = extract_sentences(f.read())

        title = extract_title_from_html(plain_file)
        print(f"Extracted title: {title}")

        plain_results = classify_sentences(plain_sentences, title, url, classifier)
        rendered_results = classify_sentences(rendered_sentences, title, url, classifier)

        # Save plain sentences
        plain_sentences_file = os.path.join(output_dir, f"{safe_url}_sentences_plain.txt")
        with open(plain_sentences_file, "w", encoding="utf-8") as f:
            for _, _, sentence, label, _ in plain_results:
                f.write(f"{sentence} (Relevant: {label})\n")

        # Save rendered sentences
        rendered_sentences_file = os.path.join(output_dir, f"{safe_url}_sentences_rendered.txt")
        with open(rendered_sentences_file, "w", encoding="utf-8") as f:
            for _, _, sentence, label, _ in rendered_results:
                f.write(f"{sentence} (Relevant: {label})\n")

        plain_relevant = sum(1 for _, _, _, label, _ in plain_results if label == "LABEL_1")
        rendered_relevant = sum(1 for _, _, _, label, _ in rendered_results if label == "LABEL_1")

        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["url", "plain_relevant", "rendered_relevant", "difference"])
            writer.writerow({
                "url": url,
                "plain_relevant": plain_relevant,
                "rendered_relevant": rendered_relevant,
                "difference": rendered_relevant - plain_relevant
            })

        for url, title, sentence, label, score in plain_results + rendered_results:
            with open(detail_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["url", "title", "sentence", "relevant", "score"])
                writer.writerow({
                    "url": url,
                    "title": title,
                    "sentence": sentence,
                    "relevant": 1 if label == "LABEL_1" else 0,
                    "score": score
                })

    print(f"Classification completed. Check {summary_file} and {detail_file}")


# Example usage
urls_to_process = [
    # "https://support.xbox.com/en-US/help/account-profile/profile/change-xbox-live-gamertag",
    # "https://www.python.org/",
    # "https://en.wikipedia.org/wiki/Web_scraping",
    # "https://en.wikipedia.org/wiki/Castle",
    # "https://www.duckduckgo.com/",

    # # dataset to validate from prod
    # "https://www.mlb.com/",
    # "https://ciscosgallery.com/collections/native-american-hat-baskets",
    # "https://zapier.com/blog/what-is-gpt/",
    
    # "https://reactormag.com/television-review-star-trek-section-31/",
    # "https://usafacts.org/data/topics/people-society/population-and-demographics/our-changing-population/",
    # "https://www.imdb.com/title/tt32252772/",
    # "https://www.wunderground.com/forecast/us/ca/oakland",
    "https://en.wikipedia.org/wiki/Aubrey_Plaza",
    # "https://nypost.com/2022/02/20/ex-teacher-who-fed-students-semen-laced-cupcakes-sentenced-to-41-years/",
    # "https://cooking.nytimes.com/68861692-nyt-cooking/950138-best-slow-cooker-recipes",
    # "https://www.webmd.com/drugs/2/drug-17765-5294/tamiflu-oral/oseltamivir-oral/details",
    # "https://www.wunderground.com/forecast/us/ca/san-francisco",
    # "https://www.webmd.com/drugs/2/drug-162440/icosapent-ethyl-oral/details",
    "https://en.wikipedia.org/wiki/Kidnapping_of_Naama_Levy",
    # "https://www.businesstoday.in/india/story/republic-day-2025-40-wishes-greetings-messages-to-send-your-friends-loved-ones-on-january-26-462049-2025-01-26",
    "https://en.wikipedia.org/wiki/List_of_awards_and_nominations_received_by_Kendrick_Lamar",
    # "https://www.bankrate.com/mortgages/mortgage-rates/",
    "https://en.wikipedia.org/wiki/Miss_Scarlet_and_The_Duke",
    "https://cooking.nytimes.com/topics/desserts",
    # "https://bobistheoilguy.com/forums/threads/stubby-pneumatic-impact-wrench.363873/",
    
    #
    # "https://www.amazon.jobs/",
    # "https://www.gettyimages.com/photos/bianca-censori?page=3",
    # "https://abcnews.go.com/",
    # "https://www.cbssports.com/nfl/scoreboard/",
    # "https://zapier.com/blog/how-does-chatgpt-work/",
    # "https://www.espn.com/nfl/scoreboard",
    # "https://yandex.com/",
    # "https://www.nfl.com/super-bowl/",
    # "https://www.foxsports.com/live-blog/nfl/bills-vs-chiefs-live-updates-analysis-highlights-from-afc-title-game",
    # "https://www.cricbuzz.com/",
    # "https://www.espn.com/nfl/schedule",
    # "https://www.thedailybeast.com/obsessed/kanye-wests-wife-bianca-censori-goes-fully-naked-at-grammys-shocks-internet/",
    # "https://www.foxsports.com/live-blog/nfl/commanders-vs-eagles-live-updates-analysis-highlights-from-nfc-title-game",
    # "https://www.nytimes.com/games/connections",
    # "https://www.espn.com/nfl/schedule/_/seasontype/2",
    # "https://abc7.com/",
    # "https://www.dictionary.com/browse/revue",
    # "https://www.ufc.com/",
    # "https://www.nytimes.com/crosswords",
    # "https://www.forbes.com/profile/elon-musk/",
    # "https://www.nytimes.com/games/strands",
    # "https://www.britannica.com/dictionary/hangdog",
    # "https://www.cnbc.com/",
    # "https://www.britannica.com/dictionary/parlay",
    # "https://www.espn.com/nfl/team/schedule/_/name/kc/kansas-city-chiefs",
    # "https://www.nbc.com/saturday-night-live",
    # "https://sports.yahoo.com/nfl/teams/washington/",
    # "https://www.myip.com/",
    # "https://ausopen.com/",
    # "https://www.cnbc.com/quotes/.SPX",
    # "https://www.10000recipe.com/en/2381158/Duck_Cheese_Quasadia?srsltid=AfmBOoqSucxUiHxRpD6uEH69Os4FVsfXQhb7XWdlzFyskE7BBJiWdxol",
    # "https://www.seatguru.com/airlines/American_Airlines/American_Airlines_Canadair_CRJ70.php",
    # "https://www.nytimes.com/puzzles/letter-boxed",
    # "https://www.streameast.soccer/streams/Soccer",
    # "https://www.buzzfeed.com/leylamohammed/kanye-west-bianca-censori-not-kicked-out-of-grammys-report",
    # "https://www.rottentomatoes.com/",
    # "https://www.fidelityworkplace.com/s/",
    # "https://www.godtube.com/watch/?v=1J22JCNU",
    # "https://news.bitcoin.com/xrp-market-update-xrp-records-massive-12-surge-is-more-upside-coming/",
    # "https://www.espn.com/soccer/standings/_/league/eng.1",
    # "https://www.espn.com/blog/buffalo-bills",
    # "https://www.peacocktv.com/",
    # "https://www.dictionary.com/browse/muck",
    # "https://ausopen.com/draws",
    # "https://www.espncricinfo.com/",
    # "https://www.pgatour.com/leaderboard",
    # "https://www.sportsgrid.com/nfl/article/do-the-chiefs-play-today-nfl-schedule-for-kansas-citys-next-game-3",
    # "https://www.uefa.com/uefachampionsleague/",
    # "https://www.cbssports.com/nfl/superbowl/",
    # "https://www.espn.com/nfl/schedule/_/week/18/year/2024/seasontype/2",
]

process_urls(urls_to_process) 
