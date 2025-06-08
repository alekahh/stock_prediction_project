import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import math

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Directories
base_path = Path(__file__).resolve().parents[1]
data_dir = base_path / "data" / "processed"
results_dir = base_path / "results" / "plots"
results_dir.mkdir(parents=True, exist_ok=True)

# Model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class AdvancedTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)

# Helper

def create_sequences(X, y, seq_length=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def evaluate_metrics(actual_returns, pred_returns, actual_prices, pred_prices):
    return {
        "Return": {
            "MSE": mean_squared_error(actual_returns, pred_returns),
            "RMSE": np.sqrt(mean_squared_error(actual_returns, pred_returns)),
            "MAE": mean_absolute_error(actual_returns, pred_returns),
            "R2": r2_score(actual_returns, pred_returns),
            "Directional_Accuracy": np.mean(np.sign(actual_returns) == np.sign(pred_returns)) * 100
        },
        "Price": {
            "MSE": mean_squared_error(actual_prices, pred_prices),
            "RMSE": np.sqrt(mean_squared_error(actual_prices, pred_prices)),
            "MAE": mean_absolute_error(actual_prices, pred_prices),
            "MAPE": np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100,
            "R2": r2_score(actual_prices, pred_prices)
        }
    }

def train_on_company(ticker):
    df = pd.read_csv(data_dir / f"{ticker}_data_with_selected_features.csv")
    df = df.sort_values("Date").reset_index(drop=True)

    features = [col for col in df.columns if col not in ['Date', 'Close', 'target']]

    split_1 = int(len(df) * 0.7)
    split_2 = int(len(df) * 0.85)

    train_df = df.iloc[:split_1].copy()
    val_df = df.iloc[split_1:split_2].copy()
    test_df = df.iloc[split_2:].copy()

    f_scaler = StandardScaler().fit(train_df[features])
    t_scaler = StandardScaler().fit(train_df[['target']])

    def prepare_split(data):
        X = f_scaler.transform(data[features])
        y = t_scaler.transform(data[['target']])
        return create_sequences(X, y)

    X_train, y_train = prepare_split(train_df)
    X_val, y_val = prepare_split(val_df)
    X_test, y_test = prepare_split(test_df)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedTransformer(feature_dim=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    best_val = float('inf')
    no_improve, patience = 0, 10
    plot_dir = results_dir / ticker
    plot_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(100):
        model.train()
        train_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in train_loader) / len(train_loader)

        model.eval()
        val_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader) / len(val_loader)

        print(f"[{ticker}] Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_model_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        scheduler.step(val_loss)

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()

    pred_returns = t_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_returns = t_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    test_index_start = test_df.index[0] + 20
    test_index_end = test_index_start + len(pred_returns)
    close_series = df['Close'].iloc[test_index_start-1:test_index_end].reset_index(drop=True)

    current_prices = close_series.iloc[:-1].values
    actual_prices = close_series.iloc[1:].values
    pred_prices = current_prices * np.exp(pred_returns)

    metrics = evaluate_metrics(actual_returns, pred_returns, actual_prices, pred_prices)

    print(f"[{ticker}] Return Metrics: {metrics['Return']}")
    print(f"[{ticker}] Price Metrics: {metrics['Price']}")

    dates = pd.to_datetime(df['Date'].iloc[test_index_start:test_index_end])
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual_prices, label='Actual')
    plt.plot(dates, pred_prices, label='Predicted', linestyle='--')
    plt.title(f"{ticker} Price Prediction\nRMSE: {metrics['Price']['RMSE']:.2f}, MAPE: {metrics['Price']['MAPE']:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig(plot_dir / f"{ticker}_price_prediction.png")
    plt.close()

if __name__ == "__main__":
    for ticker in ["qqq", "ABBV", "BABA", "COST", "EBAY", "GE", "GILD", "GLD", "GSK", "KO"]:
        train_on_company(ticker)
