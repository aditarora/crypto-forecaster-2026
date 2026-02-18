import pandas as pd

print("=== DEBUG VERSION - Preparing Dataset ===\n")

# ───── Load prices ─────
prices_df = pd.read_csv("btc_prices.csv")
print("Prices file shape:", prices_df.shape)
print("Prices columns:", prices_df.columns.tolist())
print("Prices first date:", prices_df['date'].min())
print("Prices last date :", prices_df['date'].max())
print("Sample prices:\n", prices_df.head(3))
print("Sample prices tail:\n", prices_df.tail(3))

prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
prices_df = prices_df.set_index('date')

# ───── Load sentiment ─────
sentiment_df = pd.read_csv("daily_sentiment.csv")
print("\nSentiment file shape:", sentiment_df.shape)
print("Sentiment columns:", sentiment_df.columns.tolist())
if not sentiment_df.empty:
    print("Sentiment dates:", sentiment_df['date'].tolist())
    print("Sample sentiment:\n", sentiment_df.head())
else:
    print("!!! Sentiment file is empty !!!")

sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
sentiment_df = sentiment_df.set_index('date')

# ───── Merge ─────
df = prices_df[['price']].copy()   # start minimal
df = df.join(sentiment_df[['sentiment']], how='left')

print("\nAfter join - shape:", df.shape)
print("Rows with sentiment:", df['sentiment'].notna().sum())
print("Rows without sentiment:", df['sentiment'].isna().sum())

# ───── Fill sentiment ─────
if df['sentiment'].notna().any():
    last_valid = df['sentiment'].dropna().iloc[-1]
    print(f"\nFilling ALL missing sentiment with last known value: {last_valid:.4f}")
    df['sentiment'] = df['sentiment'].fillna(last_valid)
else:
    print("\nNo sentiment values at all → filling with 0")
    df['sentiment'] = 0.0

print("After fill - missing sentiment:", df['sentiment'].isna().sum())

# ───── Add minimal features (no rolling yet) ─────
df['price_change_pct'] = df['price'].pct_change() * 100

# Only drop the very first row (NaN in pct_change)
df = df.dropna(subset=['price_change_pct'])

print("\nFinal shape before any rolling:", df.shape)
print("Final columns:", df.columns.tolist())
print("\nLast 8 rows:\n", df.tail(8))

# ───── Optional: add moving averages only if we have enough rows ─────
if len(df) >= 14:
    df['ma_7']  = df['price'].rolling(7).mean()
    df['ma_14'] = df['price'].rolling(14).mean()
    print("Added moving averages")
else:
    print("Not enough rows to compute moving averages → skipping")

# Final dropna (should only remove first 13 rows if ma added)
df = df.dropna()

print("\nFINAL DATASET shape:", df.shape)
print("FINAL columns:", df.columns.tolist())
print("\nFINAL last 5 rows:\n", df.tail(5))

df.to_csv("btc_dataset_for_ml.csv", index=True)
print("\nSaved → btc_dataset_for_ml.csv")