import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=== LSTM BTC Price Forecaster - Training ===\n")

# Load the prepared dataset
df = pd.read_csv("btc_dataset_for_ml.csv", parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"]).dt.date
df.set_index("date", inplace=True)

# Features (order matters: price is column 0 for target)
features = ["price", "sentiment", "price_change_pct", "ma_7", "ma_14"]
data = df[features].values.astype(np.float32)

print("Training data shape:", data.shape)

# Scale everything to [0,1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences: past seq_length days → predict next price
seq_length = 30

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # next price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)

if len(X) == 0:
    print(f"Error: Not enough data ({len(data)} rows) for seq_length={seq_length}")
    exit()

print(f"Created {len(X)} sequences (usable training examples)")

# Train/test split — no shuffle (time order matters!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# To PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_test  = torch.from_numpy(X_test).float()
y_test  = torch.from_numpy(y_test).float().unsqueeze(1)

# Simple LSTM model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep

model = LSTMForecaster()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 200
losses = []

print("\nTraining model...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "lstm_btc_forecaster.pth")
print("\nModel saved → lstm_btc_forecaster.pth")

# Quick test on last part of data
model.eval()
with torch.no_grad():
    test_preds = model(X_test).cpu().numpy().flatten()

# Unscale only the price predictions
def inverse_scale_price(arr):
    dummy = np.zeros((len(arr), len(features)))
    dummy[:, 0] = arr
    return scaler.inverse_transform(dummy)[:, 0]

preds_unscaled = inverse_scale_price(test_preds)
actual_unscaled = inverse_scale_price(y_test.numpy().flatten())

print("\nTest set predictions vs actual (last 5):")
for i in range(-5, 0):
    print(f"Predicted: ${preds_unscaled[i]:,.0f}   Actual: ${actual_unscaled[i]:,.0f}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss (should decrease)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("training_loss.png")
print("Loss curve saved → training_loss.png")

# ────────────────────────────────────────────────
# Forecast next 7 days
# ────────────────────────────────────────────────
print("\n=== Generating 7-day forecast ===")

model.eval()

# Take the last seq_length days as starting point
last_sequence = scaled_data[-seq_length:].copy()  # shape (30, 5)

future_days = 7
future_predictions = []

current_seq = torch.from_numpy(last_sequence).float().unsqueeze(0)  # (1, 30, 5)

with torch.no_grad():
    for day in range(future_days):
        pred_scaled = model(current_seq).item()  # predicted scaled price
        
        # Create next input row
        next_row = last_sequence[-1].copy()
        next_row[0] = pred_scaled  # update price
        
        # Shift the sequence
        last_sequence = np.vstack((last_sequence[1:], next_row))
        current_seq = torch.from_numpy(last_sequence).float().unsqueeze(0)
        
        future_predictions.append(pred_scaled)

# Unscale predictions
dummy_future = np.zeros((future_days, len(features)))
dummy_future[:, 0] = future_predictions
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

print("\nForecast for next 7 days:")
today = df.index[-1]
for i, price in enumerate(future_prices, 1):
    forecast_date = today + pd.Timedelta(days=i)
print(f"{forecast_date.strftime('%Y-%m-%d')}: ${price:,.0f}")