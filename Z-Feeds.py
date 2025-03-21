#!/usr/bin/env python3
"""
Stock-Market-LSTM Financial News Aggregator
------------------------------------------
This module collects financial news from various RSS feeds and saves them
in the Data/News directory for further analysis. It focuses on collecting data
useful for swing trading on a multi-day timeframe.
"""

import os
import json
import time
import datetime
import logging
import requests
import xmltodict
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Import utilities if available in your project
try:
    from Util import get_logger
    logger = get_logger(script_name="ZZnews")
except ImportError:
    # Set up basic logging if Util module is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("Data/logging/ZZnews.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Z-Feeds")

# Create News directory in Data folder if it doesn't exist
DATA_DIR = Path("Data")
NEWS_DIR = DATA_DIR / "News"
NEWS_DIR.mkdir(exist_ok=True, parents=True)




# Your RSS feed URLs - combined with additional financial sources



RSS_FEEDS = [
    # NYT
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml", "name": "NYT Economy", "category": "economy"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "name": "NYT Business", "category": "business"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml", "name": "NYT Politics", "category": "politics"},
    
    # Yahoo
    {"url": "https://www.yahoo.com/news/rss", "name": "Yahoo News", "category": "news"},
    {"url": "https://finance.yahoo.com/news/rssindex", "name": "Yahoo Finance", "category": "finance"},
    
    # LA Times
    {"url": "https://www.latimes.com/local/rss2.0.xml", "name": "LA Times Local", "category": "local"},
    
    # BBC
    {"url": "http://feeds.bbci.co.uk/news/business/rss.xml", "name": "BBC Business", "category": "business"},
    {"url": "http://feeds.bbci.co.uk/news/technology/rss.xml", "name": "BBC Technology", "category": "technology"},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "name": "BBC World", "category": "world"},
    {"url": "http://feeds.bbci.co.uk/news/economy/rss.xml", "name": "BBC Economy", "category": "economy"},
    
    # AP
    {"url": "https://feedx.net/rss/ap.xml", "name": "AP News", "category": "news"},
    
    # Euro News
    {"url": "https://www.euronews.com/rss", "name": "Euro News", "category": "news"},
    
    # Le Monde
    {"url": "https://www.lemonde.fr/en/rss/une.xml", "name": "Le Monde", "category": "news"},
    
    # Time
    {"url": "https://time.com/feed/", "name": "Time", "category": "news"},
    
    # Fox News
    {"url": "https://moxie.foxnews.com/google-publisher/latest.xml", "name": "Fox Latest", "category": "news"},
    {"url": "https://moxie.foxnews.com/google-publisher/world.xml", "name": "Fox World", "category": "world"},
    {"url": "https://moxie.foxnews.com/google-publisher/politics.xml", "name": "Fox Politics", "category": "politics"},
    {"url": "https://moxie.foxnews.com/google-publisher/science.xml", "name": "Fox Science", "category": "science"},
    {"url": "https://moxie.foxnews.com/google-publisher/health.xml", "name": "Fox Health", "category": "health"},
    {"url": "https://moxie.foxnews.com/google-publisher/tech.xml", "name": "Fox Tech", "category": "technology"},
    
    # Financial Times
    {"url": "https://www.ft.com/rss/home", "name": "Financial Times Home", "category": "business"},
    {"url": "https://www.ft.com/rss/world", "name": "Financial Times World", "category": "world"},
    {"url": "https://www.ft.com/rss/companies", "name": "Financial Times Companies", "category": "business"},
    
    # CNBC
    {"url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "name": "CNBC Finance", "category": "finance"},
    {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html", "name": "CNBC Economy", "category": "economy"},
    {"url": "https://www.cnbc.com/id/10001147/device/rss/rss.html", "name": "CNBC Business", "category": "business"},
    
    # Wall Street Journal
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "name": "WSJ Markets", "category": "markets"},
    {"url": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "name": "WSJ Business", "category": "business"},
    
    # MarketWatch
    {"url": "http://feeds.marketwatch.com/marketwatch/topstories/", "name": "MarketWatch Top Stories", "category": "finance"},
    {"url": "http://feeds.marketwatch.com/marketwatch/marketpulse/", "name": "MarketWatch Market Pulse", "category": "markets"},
    
    # Reuters
    {"url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", "name": "Reuters Business", "category": "business"},
    
    # Bloomberg (via Google News)
    {"url": "https://news.google.com/rss/search?q=site:bloomberg.com+finance&hl=en-US&gl=US&ceid=US:en", "name": "Bloomberg Finance via Google News", "category": "finance"},
    
    # The Economist
    {"url": "https://www.economist.com/finance-and-economics/rss.xml", "name": "Economist Finance", "category": "finance"},
    
    # Seeking Alpha
    {"url": "https://seekingalpha.com/feed.xml", "name": "Seeking Alpha", "category": "investing"},
    
    # Investing.com
    {"url": "https://www.investing.com/rss/news.rss", "name": "Investing.com News", "category": "investing"},
    {"url": "https://www.investing.com/rss/market_overview.rss", "name": "Investing.com Market Overview", "category": "markets"},
    
    # Barron's
    {"url": "https://www.barrons.com/feed/rssheadlines", "name": "Barron's Headlines", "category": "investing"},
    
    # Fortune
    {"url": "https://fortune.com/feed", "name": "Fortune", "category": "business"},
    
    # Forbes
    {"url": "https://www.forbes.com/business/feed/", "name": "Forbes Business", "category": "business"},
    {"url": "https://www.forbes.com/money/feed/", "name": "Forbes Money", "category": "finance"},
    
    # International Sources
    {"url": "https://economictimes.indiatimes.com/rssfeedsdefault.cms", "name": "Economic Times India", "category": "business"},
    {"url": "https://www.afr.com/rss/latest-news", "name": "Australian Financial Review", "category": "business"},
]





def fetch_rss(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch and process an RSS feed
    
    Args:
        source: Source information including URL
        
    Returns:
        List of extracted articles
    """
    url = source["url"]
    name = source["name"]
    articles = []
    
    try:
        logger.info(f"Fetching: {name} ({url})")
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code} for {name}")
            return articles
            
        data = xmltodict.parse(response.content)
        
        # Most RSS feeds follow this structure
        if 'rss' in data and 'channel' in data['rss']:
            channel = data['rss']['channel']
            items = channel.get('item', [])
            
            # Handle case when there's only one item
            if not isinstance(items, list):
                items = [items]
                
            for item in items:
                # Skip if item is None or not a dict
                if not item or not isinstance(item, dict):
                    continue
                    
                # Extract article information
                article = {
                    "title": item.get('title', ''),
                    "description": item.get('description', ''),
                    "link": item.get('link', ''),
                    "pub_date": item.get('pubDate', ''),
                    "source_name": source['name'],
                    "source_url": source['url'],
                    "category": source['category'],
                    "fetch_time": datetime.datetime.now().isoformat()
                }
                
                # Add guid if available
                if 'guid' in item:
                    if isinstance(item['guid'], dict):
                        article["guid"] = item['guid'].get('#text', '')
                    else:
                        article["guid"] = item['guid']
                
                # Add content if available
                if 'content:encoded' in item:
                    article["content"] = item['content:encoded']
                    
                # Add the article to our list
                articles.append(article)
        
        elif 'feed' in data:  # Atom feed format
            feed = data['feed']
            entries = feed.get('entry', [])
            
            # Handle case when there's only one entry
            if not isinstance(entries, list):
                entries = [entries]
                
            for entry in entries:
                # Skip if entry is None or not a dict
                if not entry or not isinstance(entry, dict):
                    continue
                
                # Extract link
                link = entry.get('link', '')
                if isinstance(link, list):
                    for l in link:
                        if l.get('@rel') == 'alternate':
                            link = l.get('@href', '')
                            break
                elif isinstance(link, dict):
                    link = link.get('@href', '')
                
                # Extract article information
                article = {
                    "title": entry.get('title', ''),
                    "description": entry.get('summary', ''),
                    "link": link,
                    "pub_date": entry.get('updated', ''),
                    "source_name": source['name'],
                    "source_url": source['url'],
                    "category": source['category'],
                    "fetch_time": datetime.datetime.now().isoformat()
                }
                
                # Add id if available
                if 'id' in entry:
                    article["guid"] = entry['id']
                
                # Add content if available
                if 'content' in entry:
                    if isinstance(entry['content'], dict):
                        article["content"] = entry['content'].get('#text', '')
                    else:
                        article["content"] = entry['content']
                
                # Add the article to our list
                articles.append(article)
        
        logger.info(f"Found {len(articles)} articles from {name}")
        return articles
        
    except Exception as e:
        logger.error(f"Error processing {name}: {str(e)}")
        return articles

def save_articles(articles: List[Dict[str, Any]], filename: str) -> None:
    """Save articles to a JSON file
    
    Args:
        articles: List of articles to save
        filename: Filename to save the articles
    """
    filepath = NEWS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Saved {len(articles)} articles to {filepath}")

def save_articles_parquet(articles: List[Dict[str, Any]], filepath: str) -> None:
    """Save articles to a Parquet file
    
    Args:
        articles: List of articles to save
        filepath: Path to save the Parquet file
    """
    # Convert to pandas DataFrame
    df = pd.DataFrame(articles)
    
    # Save to Parquet
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(articles)} articles to {filepath}")

def clean_article_data(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and normalize article data
    
    Args:
        articles: List of articles to clean
        
    Returns:
        List of cleaned articles
    """
    cleaned_articles = []
    
    for article in articles:
        # Remove HTML tags from description using pandas
        if 'description' in article and article['description']:
            try:
                article['description'] = pd.Series(article['description']).str.replace(r'<.*?>', '', regex=True)[0]
            except:
                pass
        
        # Normalize pub_date formats
        if 'pub_date' in article and article['pub_date']:
            try:
                # Handle common date formats
                for fmt in [
                    '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
                    '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
                    '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
                    '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
                    '%Y-%m-%d %H:%M:%S',         # Simple format
                ]:
                    try:
                        dt = datetime.datetime.strptime(article['pub_date'], fmt)
                        article['pub_date_iso'] = dt.isoformat()
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep the original
                pass
        
        cleaned_articles.append(article)
    
    return cleaned_articles

def fetch_all_feeds(max_workers: int = 10) -> List[Dict[str, Any]]:
    """Fetch all RSS feeds in parallel
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of all extracted articles
    """
    all_articles = []
    
    logger.info(f"Fetching {len(RSS_FEEDS)} feeds with {max_workers} parallel workers...")
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_rss, RSS_FEEDS))
    
    # Combine all results
    for articles in results:
        all_articles.extend(articles)
    
    logger.info(f"Fetched a total of {len(all_articles)} articles")
    return all_articles

def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate articles based on title or link
    
    Args:
        articles: List of articles to deduplicate
        
    Returns:
        List of deduplicated articles
    """
    unique_articles = []
    seen_titles = set()
    seen_links = set()
    
    for article in articles:
        title = article.get('title', '').strip()
        link = article.get('link', '').strip()
        
        # Skip if we've seen this title or link before
        if title and title in seen_titles:
            continue
        if link and link in seen_links:
            continue
        
        # Add to our unique list
        unique_articles.append(article)
        
        # Remember we've seen this title and link
        if title:
            seen_titles.add(title)
        if link:
            seen_links.add(link)
    
    logger.info(f"Removed {len(articles) - len(unique_articles)} duplicate articles")
    return unique_articles

def main():
    """Main function to run the RSS aggregator"""
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    
    logger.info("Starting RSS Feed Aggregator...")
    logger.info(f"Found {len(RSS_FEEDS)} RSS feeds to process")
    
    # Fetch all feeds in parallel
    articles = fetch_all_feeds(max_workers=15)
    
    # Clean the article data
    articles = clean_article_data(articles)
    
    # Deduplicate articles
    articles = deduplicate_articles(articles)
    
    # Save the articles to JSON
    json_filename = f"financial_news_{timestamp}.json"
    save_articles(articles, json_filename)
    
    # Save to Parquet (integrates better with your system)
    parquet_path = NEWS_DIR / f"financial_news_{timestamp}.parquet"
    save_articles_parquet(articles, parquet_path)
    
    # Also maintain a latest copy for easy access
    latest_parquet_path = NEWS_DIR / "latest_financial_news.parquet"
    save_articles_parquet(articles, latest_parquet_path)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Done! Completed in {elapsed_time:.2f} seconds")
    logger.info(f"- Processed {len(RSS_FEEDS)} RSS feeds")
    logger.info(f"- Fetched {len(articles)} unique articles")
    logger.info(f"- Saved to {NEWS_DIR / json_filename}")
    logger.info(f"- Saved to {parquet_path}")
    logger.info(f"- Updated {latest_parquet_path}")

if __name__ == "__main__":
    main()