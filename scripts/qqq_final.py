import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================
# 1. LOAD AND PREPARE DATA
# =====================
base_path = Path(__file__).resolve().parents[1]
processed_data_dir = base_path / "data" / "processed"
qqq_data = pd.read_csv(processed_data_dir / "qqq_data_with_selected_features.csv", parse_dates=['Date'])
qqq_data = qqq_data.sort_values('Date').reset_index(drop=True)

# Define feature groups
financial_features = [
    'price_to_sma20', 'return_5d', 'volatility_5d', 'return_1d',
    'price_to_sma5', 'gap', 'volatility_20d', 'volume_ratio', 'volume_change',
    'sma_20', 'sma_5'
]

macro_features = [
    'sp500_return', 'oil_price', 'industrial_output'
]

sentiment_features = [
    'sentiment_3d_avg', 'sentiment_5d_avg', 'Scaled_sentiment', 'Sentiment_gpt'
]

feature_groups = {
    "Financial Only": financial_features,
    "Financial + Macro": financial_features + macro_features,
    "All Features": financial_features + macro_features + sentiment_features
}

# =====================
# 2. MODEL ARCHITECTURE
# =====================
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

class QQQTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)

# =====================
# 3. TRAINING AND EVALUATION FUNCTION
# =====================
def train_and_evaluate(feature_cols, group_name):
    print(f"\n=== Training with {group_name} ({len(feature_cols)} features) ===")
    
    # Data preparation
    train_size = int(len(qqq_data) * 0.7)
    val_size = int(len(qqq_data) * 0.15)
    
    train_data = qqq_data.iloc[:train_size].copy()
    val_data = qqq_data.iloc[train_size:train_size+val_size].copy()
    test_data = qqq_data.iloc[train_size+val_size:].copy()
    
    # Scaling
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_train_scaled = feature_scaler.fit_transform(train_data[feature_cols])
    y_train_scaled = target_scaler.fit_transform(train_data[['target']])
    
    X_val_scaled = feature_scaler.transform(val_data[feature_cols])
    y_val_scaled = target_scaler.transform(val_data[['target']])
    X_test_scaled = feature_scaler.transform(test_data[feature_cols])
    y_test_scaled = target_scaler.transform(test_data[['target']])
    
    # Sequence creation
    def create_sequences(X, y, seq_length=20):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)
    
    # Convert to tensors
    X_train = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val = torch.tensor(y_val_seq, dtype=torch.float32)
    X_test = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test = torch.tensor(y_test_seq, dtype=torch.float32)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QQQTransformer(feature_dim=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Training
    best_val_loss = float('inf')
    patience = 10
    no_improve_epochs = 0
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.to(device))
            loss = criterion(outputs, batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device))
                val_loss += criterion(outputs, batch_y.to(device)).item()
        
        scheduler.step(val_loss)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), f'best_model_{group_name.replace(" ", "_")}.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluation
    model.load_state_dict(torch.load(f'best_model_{group_name.replace(" ", "_")}.pth'))
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
    
    pred_returns = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_returns = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
    # Return metrics
    return_metrics = {
        'MSE': float(mean_squared_error(actual_returns, pred_returns)),
        'RMSE': float(np.sqrt(mean_squared_error(actual_returns, pred_returns))),
        'MAE': float(mean_absolute_error(actual_returns, pred_returns)),
        'R2': float(r2_score(actual_returns, pred_returns)),
        'Directional_Accuracy': float(np.mean(np.sign(actual_returns) == np.sign(pred_returns))) * 100
    }
    
    # Price metrics
    current_prices = test_data['Close'].iloc[20-1:20-1+len(pred_returns)].values
    pred_prices = current_prices * np.exp(pred_returns)
    actual_prices = test_data['Close'].iloc[20:20+len(pred_returns)].values
    
    price_metrics = {
        'MSE': float(mean_squared_error(actual_prices, pred_prices)),
        'RMSE': float(np.sqrt(mean_squared_error(actual_prices, pred_prices))),
        'MAE': float(mean_absolute_error(actual_prices, pred_prices)),
        'MAPE': float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices))) * 100,
        'R2': float(r2_score(actual_prices, pred_prices))
    }
    
    # Plotting
    test_dates = test_data['Date'].iloc[20:20+len(pred_returns)].values
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, actual_prices, label='Actual', color='blue')
    plt.plot(test_dates, pred_prices, label='Predicted', color='red', linestyle='--')
    plt.title(f'QQQ Price Prediction ({group_name})\nRMSE: {price_metrics["RMSE"]:.2f}, MAPE: {price_metrics["MAPE"]:.2f}%')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'qqq_prediction_{group_name.replace(" ", "_")}.png')
    plt.close()
    
    return return_metrics, price_metrics

# =====================
# 4. RUN EXPERIMENTS FOR ALL FEATURE GROUPS
# =====================
results = {}
for group_name, features in feature_groups.items():
    return_metrics, price_metrics = train_and_evaluate(features, group_name)
    results[group_name] = {
        'Return Metrics': return_metrics,
        'Price Metrics': price_metrics
    }
    print(f"\nResults for {group_name}:")
    print("Return Prediction:", return_metrics)
    print("Price Prediction:", price_metrics)

# =====================
# 5. COMPARE RESULTS
# =====================
print("\n=== Final Comparison ===")
for group_name, metrics in results.items():
    print(f"\n{group_name}:")
    print(f"  Return R²: {metrics['Return Metrics']['R2']:.4f}")
    print(f"  Price R²: {metrics['Price Metrics']['R2']:.4f}")
    print(f"  Directional Accuracy: {metrics['Return Metrics']['Directional_Accuracy']:.2f}%")
    print(f"  Price MAPE: {metrics['Price Metrics']['MAPE']:.2f}%")
