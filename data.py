import requests
import pandas as pd
from datetime import datetime
import feedparser
from transformers import pipeline
import torch
import time

# ────────────────────────────────────────────────
# 1. Fetch Historical Bitcoin Prices from CoinGecko (no API key needed)
# ────────────────────────────────────────────────
def get_bitcoin_prices(days=90, retries=3):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            prices = data.get("prices", [])
            if not prices:
                print("No price data returned from CoinGecko.")
                return None
                
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            df = df.drop("timestamp", axis=1)
            df.set_index("date", inplace=True)
            print(f"Successfully fetched {len(df)} days of BTC prices.")
            return df
        except Exception as e:
            print(f"Price fetch attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print("All retries failed for prices.")
                return None

# ────────────────────────────────────────────────
# 2. Fetch Recent Crypto News from Cointelegraph RSS (free, no key)
# ────────────────────────────────────────────────
def get_crypto_news_from_rss(limit=30, keywords=["bitcoin", "btc"]):
    rss_url = "https://cointelegraph.com/rss"
    try:
        feed = feedparser.parse(rss_url)
        if feed.bozo:
            print("RSS parse error:", feed.bozo_exception)
            return None
        
        entries = feed.entries[:limit]
        data = []
        for entry in entries:
            title_lower = entry.get("title", "").lower()
            # Filter for Bitcoin-relevant news (remove filter [] for all crypto news)
            if any(kw in title_lower for kw in keywords):
                data.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "pubDate": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "source": "Cointelegraph"
                })
        
        if not data:
            print("No Bitcoin-related news found. Try removing keyword filter.")
            return None
        
        df = pd.DataFrame(data)
        df["pubDate"] = pd.to_datetime(df["pubDate"])
        df["date"] = df["pubDate"].dt.date
        print(f"Fetched {len(df)} Bitcoin-related news items from Cointelegraph RSS.")
        return df
    except Exception as e:
        print(f"Error fetching RSS news: {e}")
        return None

# ────────────────────────────────────────────────
# 3. Sentiment Analysis using Hugging Face (free, local model)
# ────────────────────────────────────────────────
sentiment_analyzer = None

def get_sentiment(text):
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("Loading Hugging Face sentiment model... (downloads ~500MB first time only)")
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        sentiment_analyzer = pipeline("sentiment-analysis", device=device)
    
    if not text or pd.isna(text):
        return 0.0
    
    # Truncate long text to avoid model limit
    result = sentiment_analyzer(text[:512])[0]
    score = result["score"]
    if result["label"] == "NEGATIVE":
        score = -score
    return round(score, 4)  # e.g. 0.85 positive, -0.72 negative

def add_sentiment_to_news(news_df):
    if news_df is None or news_df.empty:
        print("No news data to analyze.")
        return None
    
    print("Calculating sentiment on news titles & descriptions...")
    # Combine title + description for better context
    combined_text = news_df["title"] + " " + news_df["description"].fillna("")
    news_df["sentiment"] = combined_text.apply(get_sentiment)
    
    # Aggregate daily average sentiment
    daily_sentiment = news_df.groupby("date")["sentiment"].mean().reset_index()
    daily_sentiment["sentiment"] = daily_sentiment["sentiment"].round(4)
    print("Daily average sentiment calculated.")
    return daily_sentiment

# ────────────────────────────────────────────────
# MAIN: Run everything and save CSVs
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Crypto News Sentiment Forecaster - Data Fetcher ===\n")
    
    # Step 1: Prices
    btc_df = get_bitcoin_prices(days=90)
    if btc_df is not None:
        print("\nRecent BTC Prices (first 5 rows):")
        print(btc_df.head())
        btc_df.to_csv("btc_prices.csv")
        print("Saved prices → btc_prices.csv")
    
    # Step 2: News
    news_df = get_crypto_news_from_rss(limit=30)
    if news_df is not None:
        print("\nRecent BTC News (first 5 rows):")
        print(news_df[["date", "title", "source"]].head())
        news_df.to_csv("btc_news.csv")
        print("Saved news → btc_news.csv")
        
        # Step 3: Sentiment
        daily_sent_df = add_sentiment_to_news(news_df)
        if daily_sent_df is not None:
            print("\nDaily Sentiment (last 10 days):")
            print(daily_sent_df.tail(10))
            daily_sent_df.to_csv("daily_sentiment.csv")
            print("Saved daily sentiment → daily_sentiment.csv")
    
    print("\nDone! Check your folder for CSVs. Next: Merge data & train LSTM.")