import torch
torch.set_default_device('cpu')

import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import time
import feedparser
from transformers import pipeline

# Auto-refresh every 5 minutes
AUTO_REFRESH_MINUTES = 5
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > AUTO_REFRESH_MINUTES * 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ────────────────────────────────────────────────
# LSTM Model Definition
# ────────────────────────────────────────────────
class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = LSTMForecaster()
    try:
        model.load_state_dict(torch.load("lstm_btc_forecaster.pth", map_location="cpu"))
        model.eval()
    except:
        st.warning("Model file not found or failed to load.")
        return None
    return model

model = load_model()

# ────────────────────────────────────────────────
# Per-coin sentiment function
# ────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def get_coin_sentiment(coin_name, symbol):
    rss_url = "https://cointelegraph.com/rss"
    feed = feedparser.parse(rss_url)
    
    if feed.bozo:
        return 0.0, []
    
    relevant_articles = []
    keywords = [coin_name.lower(), symbol.lower()]
    
    for entry in feed.entries[:30]:
        title = entry.get("title", "").lower()
        summary = entry.get("summary", "").lower()
        text = title + " " + summary
        
        if any(kw in text for kw in keywords):
            relevant_articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", "")
            })
    
    if not relevant_articles:
        return 0.0, []
    
    if 'sentiment_pipe' not in st.session_state:
        with st.spinner("Loading AI sentiment model (one-time download, ~500MB)..."):
            st.session_state.sentiment_pipe = pipeline("sentiment-analysis")
    
    pipe = st.session_state.sentiment_pipe
    
    scores = []
    for art in relevant_articles:
        text = art["title"] + " " + art["summary"]
        if text.strip():
            result = pipe(text[:512])[0]
            score = result["score"]
            if result["label"] == "NEGATIVE":
                score = -score
            scores.append(score)
    
    avg_sentiment = np.mean(scores) if scores else 0.0
    return round(avg_sentiment, 4), relevant_articles[:5]

# ────────────────────────────────────────────────
# Sidebar – Verification Links
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Verify Data Sources")
    st.markdown("Cross-check prices, rates & news independently")
    
    if 'coin_id' in locals():
        st.markdown(f"**Price & Chart** → [CoinGecko – {selected_name}](https://www.coingecko.com/en/coins/{coin_id})")
    st.markdown("→ [CoinMarketCap](https://coinmarketcap.com)")
    
    st.markdown("**USD → INR Rate** → [Google](https://www.google.com/search?q=1+USD+to+INR)")
    st.markdown("→ [Xe.com](https://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=INR)")
    
    st.markdown("**Crypto News** → [Cointelegraph](https://cointelegraph.com)")
    st.markdown("→ [CoinDesk](https://www.coindesk.com)")
    
    st.caption("Last refresh: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M IST"))

# ────────────────────────────────────────────────
# Main UI – Beginner Guide Button
# ────────────────────────────────────────────────
st.title("Crypto Sentiment Forecaster")

if st.button("📖 How to Use This App (Beginner Guide)", type="secondary", help="Click to read simple instructions"):
    with st.popover("Beginner Guide – How to Use This App", width=700):
        st.markdown("""
        **Welcome!** This app helps you see crypto prices, news mood, and price predictions.

        **Simple steps:**
        1. Choose a cryptocurrency (Bitcoin, Ethereum, etc.)
        2. See current price, 24h/7d change, and chart
        3. Read **Sentiment Score** → shows if recent news is positive or negative
        4. Click "Generate 7-Day Forecast" → see what the model thinks will happen next
        5. Check "Backtesting" → see how accurate the model was in the past

        **Important for beginners:**
        - This is **not financial advice** — just a learning project
        - Forecasts are computer guesses — they can be wrong
        - Sentiment score comes from news headlines — not perfect, but shows general mood
        - Always check real prices on CoinGecko or other trusted sites
        """)

crypto_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano",
    "XRP": "xrp",
    "Binance Coin": "binancecoin",
    "Polkadot": "polkadot",
    "Chainlink": "chainlink",
    "Avalanche": "avalanche-2",
    "Shiba Inu": "shiba-inu",
}

selected_name = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
coin_id = crypto_options[selected_name]

st.caption(f"Selected: **{selected_name}** (CoinGecko ID: {coin_id})")

# Currency toggle
show_inr = st.checkbox("Show prices in INR (₹)", value=True)

# Fetch live USD/INR rate
@st.cache_data(ttl=900)
def get_usd_to_inr_rate():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=usd&vs_currencies=inr"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.json()["usd"]["inr"]
    except:
        return 90.8

usd_to_inr = get_usd_to_inr_rate()
currency_symbol = "₹" if show_inr else "$"
st.caption(f"1 USD ≈ {usd_to_inr:.2f} INR (live rate)")

# Load data
@st.cache_data(ttl=1800)
def load_crypto_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=90&interval=daily"
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return pd.DataFrame()
        
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df.drop("timestamp", axis=1, inplace=True)
        df.set_index("date", inplace=True)

        if volumes and len(volumes) == len(prices):
            df["volume"] = [v[1] for v in volumes]
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        return pd.DataFrame()

    df["sentiment"] = -0.5 if "bitcoin" in coin_id else 0.0

    df["price_change_pct"] = df["price"].pct_change() * 100
    if len(df) >= 14:
        df["ma_7"] = df["price"].rolling(7).mean()
        df["ma_14"] = df["price"].rolling(14).mean()
    df.dropna(inplace=True)
    return df

df_usd = load_crypto_data(coin_id)

if df_usd.empty:
    st.error("No data available for this cryptocurrency.")
    st.stop()

# Convert to INR if selected
if show_inr:
    df = df_usd.copy()
    price_cols = [c for c in ["price", "ma_7", "ma_14"] if c in df.columns]
    for col in price_cols:
        df[col] = df[col] * usd_to_inr
else:
    df = df_usd

# Price change indicators
if len(df) >= 7:
    latest_price = df["price"].iloc[-1]
    price_24h_ago = df["price"].iloc[-2] if len(df) >= 2 else latest_price
    price_7d_ago = df["price"].iloc[-8] if len(df) >= 8 else latest_price
    change_24h = ((latest_price - price_24h_ago) / price_24h_ago) * 100
    change_7d = ((latest_price - price_7d_ago) / price_7d_ago) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"{currency_symbol}{latest_price:,.2f}")
    with col2:
        st.metric("24h Change", f"{change_24h:+.2f}%")
    with col3:
        st.metric("7d Change", f"{change_7d:+.2f}%")

# Timeframe selector
timeframe = st.selectbox("Chart Timeframe", ["All", "90 days", "30 days", "7 days"], index=0)

if timeframe == "7 days":
    chart_df = df.tail(7)
elif timeframe == "30 days":
    chart_df = df.tail(30)
elif timeframe == "90 days":
    chart_df = df.tail(90)
else:
    chart_df = df

# Historical chart with volume
st.subheader(f"{selected_name} Price & Volume")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df["price"],
    mode="lines",
    name=f"Price ({currency_symbol})",
    line=dict(color="#00CC96")
))
if "volume" in chart_df.columns:
    fig_hist.add_trace(go.Bar(
        x=chart_df.index,
        y=chart_df["volume"],
        name="Volume (USD)",
        yaxis="y2",
        opacity=0.3,
        marker_color="#636EFA"
    ))
fig_hist.update_layout(
    xaxis_title="Date",
    yaxis_title=f"Price ({currency_symbol})",
    yaxis2=dict(title="Volume (USD)", overlaying="y", side="right"),
    template="plotly_dark",
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig_hist, width='stretch')

# Per-coin sentiment with beginner explanation
st.subheader("News Sentiment – How People Feel About This Coin")

with st.expander("What is Sentiment Score? (Click to learn)", expanded=False):
    st.markdown("""
    **Sentiment score** shows if recent news about this coin is positive, negative, or neutral.

    - **+0.5 to +1.0** = Good news — people are excited (bullish)
    - **0.0** = Neutral — no strong opinion
    - **-0.5 to -1.0** = Bad news — people are worried (bearish)

    Example:
    - "Bitcoin hits new record high!" → positive score (+0.85)
    - "Ethereum network has big problems" → negative score (-0.72)

    This score is calculated by AI reading recent news headlines and summaries.
    """)

coin_name = selected_name
symbol = selected_name.split()[-1].lower() if " " in selected_name else selected_name.lower()

with st.spinner("Reading recent news and understanding the mood..."):
    sentiment_score, articles = get_coin_sentiment(coin_name, symbol)

st.metric(
    label="Current Sentiment Score",
    value=f"{sentiment_score:.4f}",
    help="Ranges from -1.0 (very negative) to +1.0 (very positive)"
)

if sentiment_score > 0.1:
    st.success("Recent news is mostly positive → people seem excited!")
elif sentiment_score < -0.1:
    st.warning("Recent news is mostly negative → people seem worried.")
else:
    st.info("Recent news is neutral → no strong feelings.")

if articles:
    st.caption(f"Based on {len(articles)} recent articles mentioning {coin_name}")
    with st.expander("See the news headlines the AI read"):
        for art in articles:
            st.markdown(f"**{art['published']}** – [{art['title']}]({art['link']})")
else:
    st.caption(f"No recent news found specifically mentioning {coin_name}.")

# Forecast with beginner tip
st.markdown("---")
st.info("""
**Beginner tip about the Forecast:**
- This is **not magic** — it's a computer program (called LSTM) that looks at past prices and tries to guess the future.
- The number with **±** shows how confident the model is (smaller ± = more sure, bigger ± = less sure).
- Always check real prices on CoinGecko — don't use this to buy or sell coins!
""")

if st.button("Generate 7-Day Forecast", type="primary"):
    if model is None:
        st.error("No trained model available.")
    elif len(df_usd) < 30:
        st.error("Not enough data for forecast.")
    else:
        with st.spinner("Forecasting with uncertainty (Monte Carlo dropout)..."):
            seq_length = 30
            avail_features = [f for f in ["price", "sentiment", "price_change_pct", "ma_7", "ma_14"] if f in df_usd.columns]
            scaler_local = MinMaxScaler().fit(df_usd[avail_features])
            last_sequence = scaler_local.transform(df_usd.tail(seq_length)[avail_features])

            model.train()
            N_SAMPLES = 30

            future_means = []
            future_stds = []

            current_seq = torch.from_numpy(last_sequence).float().unsqueeze(0)

            for step in range(7):
                sample_preds = []
                for _ in range(N_SAMPLES):
                    with torch.no_grad():
                        pred = model(current_seq).item()
                    sample_preds.append(pred)

                mean_pred = np.mean(sample_preds)
                std_pred = np.std(sample_preds)

                future_means.append(mean_pred)
                future_stds.append(std_pred)

                next_row = last_sequence[-1].copy()
                next_row[0] = mean_pred
                last_sequence = np.vstack((last_sequence[1:], next_row))
                current_seq = torch.from_numpy(last_sequence).float().unsqueeze(0)

            dummy_means = np.zeros((7, len(avail_features)))
            dummy_means[:, 0] = future_means
            future_prices_usd = scaler_local.inverse_transform(dummy_means)[:, 0]

            price_scale_factor = scaler_local.scale_[0]
            uncertainties_usd = np.array(future_stds) * price_scale_factor

            future_prices = future_prices_usd * usd_to_inr if show_inr else future_prices_usd
            uncertainties = uncertainties_usd * usd_to_inr if show_inr else uncertainties_usd

        st.subheader(f"7-Day {selected_name} Forecast with Uncertainty")
        forecast_dates = [df_usd.index[-1] + timedelta(days=i+1) for i in range(7)]
        forecast_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            f"Predicted Price ({currency_symbol})": [f"{p:,.0f} ± {u:,.0f}" for p, u in zip(future_prices, uncertainties)]
        })

        st.dataframe(
            forecast_df,
            hide_index=True,
            use_container_width=True
        )

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df.index[-40:],
            y=df["price"][-40:],
            mode="lines",
            name="Historical",
            line=dict(color="#00CC96")
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=future_prices,
            mode="lines+markers",
            name="Forecast (mean)",
            line=dict(color="#FF6B6B")
        ))
        upper = future_prices + uncertainties
        lower = future_prices - uncertainties
        fig_forecast.add_trace(go.Scatter(
            x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='± Uncertainty',
            showlegend=True
        ))
        fig_forecast.update_layout(
            title="Forecast with Uncertainty Band",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
            template="plotly_dark",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, width='stretch')

# Backtesting with beginner explanation
st.markdown("---")
with st.expander("What is Backtesting? (Click to learn)", expanded=False):
    st.markdown("""
    **Backtesting** = checking how good the model was at predicting prices in the past.

    We hide some old data → ask the model to guess what happened next → compare with real prices.

    - **Lower error** = model was more accurate in the past
    - **Higher error** = model still needs improvement

    This helps us know if we can trust the future forecast.
    """)

st.subheader("Backtesting: Model Performance on Past Data")
if model is None:
    st.info("No model loaded – backtesting unavailable.")
elif len(df_usd) < 60:
    st.info("Not enough data for meaningful backtesting.")
else:
    with st.spinner("Running backtest..."):
        seq_length = 30
        avail_features = [f for f in ["price", "sentiment", "price_change_pct", "ma_7", "ma_14"] if f in df_usd.columns]
        scaler_bt = MinMaxScaler().fit(df_usd[avail_features])
        scaled = scaler_bt.transform(df_usd[avail_features])

        actuals = []
        preds = []

        for i in range(seq_length, len(scaled) - 1):
            seq = scaled[i-seq_length:i]
            actual_next = scaled[i, 0]
            
            input_seq = torch.from_numpy(seq).float().unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model(input_seq).item()
            
            actuals.append(actual_next)
            preds.append(pred_scaled)

        dummy_actual = np.zeros((len(actuals), len(avail_features)))
        dummy_actual[:, 0] = actuals
        actual_prices = scaler_bt.inverse_transform(dummy_actual)[:, 0]

        dummy_pred = np.zeros((len(preds), len(avail_features)))
        dummy_pred[:, 0] = preds
        pred_prices = scaler_bt.inverse_transform(dummy_pred)[:, 0]

        mae = np.mean(np.abs(actual_prices - pred_prices))
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100 if np.all(actual_prices != 0) else 0

        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error (USD)", f"${mae:,.2f}", help="Average difference between predicted and real prices")
        col2.metric("Mean Absolute % Error", f"{mape:.2f}%", help="Average percentage error")

        dates = df_usd.index[seq_length:len(scaled)-1]
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=dates, y=actual_prices, mode="lines", name="Actual Price"))
        fig_bt.add_trace(go.Scatter(x=dates, y=pred_prices, mode="lines", name="Model Predicted", line=dict(dash="dash")))
        fig_bt.update_layout(title="Backtest: Actual vs Predicted Prices", template="plotly_dark")
        st.plotly_chart(fig_bt, width='stretch')

# Footer
st.caption(f"Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')}")
st.caption("Forecasts are approximate (model trained on BTC data) • Data from CoinGecko • Live USD/INR rate fetched dynamically")
st.caption(f"Auto-refreshing every {AUTO_REFRESH_MINUTES} minutes • Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S IST')}")