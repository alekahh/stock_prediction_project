import os
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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================
# 1. DATA LOADING (identical to transformer)
# =====================
base_path = Path(__file__).resolve().parents[1]
raw_data_dir = base_path / "data" / "raw"
processed_data_dir = base_path / "data" / "processed"

# Load datasets
dsf_jnj = pd.read_csv(raw_data_dir / "dsf_jnj.csv", parse_dates=['date'])
msenames = pd.read_csv(raw_data_dir / "msenames.csv", parse_dates=['NAMEDT'])

# =====================
# 2. DATA PREPARATION (identical to transformer)
# =====================
# Filter for PERMNO = 22111 (JNJ)
dsf_jnj = dsf_jnj[dsf_jnj['PERMNO'] == 22111]
df = dsf_jnj.merge(msenames, on='PERMNO', how='left')
df = df[['date', 'RET', 'VOL', 'PRC']].sort_values('date').reset_index(drop=True)

# Handle missing values
df['RET'] = df['RET'].fillna(0)
df = df.dropna()

# Train/val/test split (identical to transformer)
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train_data = df[:train_size].copy()
val_data = df[train_size:train_size+val_size].copy()
test_data = df[train_size+val_size:].copy()

# Feature engineering (identical to transformer)
def create_simple_features(data):
    data = data.copy()
    data['log_return'] = np.log(data['PRC'] / data['PRC'].shift(1))
    data['log_vol'] = np.log(data['VOL'] / data['VOL'].shift(1))
    data['target'] = data['log_return'].shift(-1)  # Next day's return
    return data.dropna()

train_data = create_simple_features(train_data)
val_data = create_simple_features(val_data)
test_data = create_simple_features(test_data)

feature_cols = ['log_return', 'log_vol']

# Scaling (identical to transformer)
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(train_data[feature_cols])
y_train_scaled = target_scaler.fit_transform(train_data[['target']])

X_val_scaled = feature_scaler.transform(val_data[feature_cols])
y_val_scaled = target_scaler.transform(val_data[['target']])
X_test_scaled = feature_scaler.transform(test_data[feature_cols])
y_test_scaled = target_scaler.transform(test_data[['target']])

# Sequence creation (identical to transformer)
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)

# Convert to tensors (identical to transformer)
X_train = torch.tensor(X_train_seq, dtype=torch.float32)
y_train = torch.tensor(y_train_seq, dtype=torch.float32)
X_val = torch.tensor(X_val_seq, dtype=torch.float32)
y_val = torch.tensor(y_val_seq, dtype=torch.float32)
X_test = torch.tensor(X_test_seq, dtype=torch.float32)
y_test = torch.tensor(y_test_seq, dtype=torch.float32)

# =====================
# 3. LSTM MODEL DEFINITION
# =====================
class StockLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# =====================
# 4. TRAINING SETUP
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Create results directory
results_dir = base_path / "results"
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# =====================
# 5. TRAINING LOOP
# =====================
def train_model(model, train_loader, val_loader, epochs=50, patience=5):
    best_val_loss = float('inf')
    no_improve = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.to(device))
            loss = criterion(outputs, batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device))
                val_loss += criterion(outputs, batch_y.to(device)).item()
        
        # Update learning rate
        scheduler.step(val_loss / len(val_loader))
        
        # Store metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        print(f'Epoch {epoch+1}/{epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}')
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), plots_dir / "best_lstm.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return history

history = train_model(model, train_loader, val_loader)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['train'], label='Training Loss')
plt.plot(history['val'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig(plots_dir / "lstm_training_history.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# 6. EVALUATION
# =====================
model.load_state_dict(torch.load(plots_dir / "best_lstm.pth"))
model.eval()

with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()

# Inverse scaling
pred_returns = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
actual_returns = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Calculate metrics
print(f"\nReturn Prediction Results:")
print(f"MSE: {mean_squared_error(actual_returns, pred_returns):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual_returns, pred_returns)):.6f}")
print(f"MAE: {mean_absolute_error(actual_returns, pred_returns):.6f}")
print(f"R²: {r2_score(actual_returns, pred_returns):.4f}")
print(f"Directional Accuracy: {np.mean(np.sign(actual_returns) == np.sign(pred_returns)) * 100:.2f}%")

# Price prediction
current_prices = test_data['PRC'].iloc[10-1:10-1+len(pred_returns)].values
pred_prices = current_prices * np.exp(pred_returns)
actual_prices = test_data['PRC'].iloc[10:10+len(pred_returns)].values

print(f"\nPrice Prediction Results:")
print(f"MSE: {mean_squared_error(actual_prices, pred_prices):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual_prices, pred_prices)):.4f}")
print(f"MAE: {mean_absolute_error(actual_prices, pred_prices):.4f}")
print(f"MAPE: {np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100:.2f}%")
print(f"R²: {r2_score(actual_prices, pred_prices):.4f}")

# =====================
# 7. PRICE MOVEMENT VISUALIZATION
# =====================
test_dates = test_data['date'].iloc[10:10+len(pred_returns)].values

# Create figure
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted Prices
plt.subplot(2, 1, 1)
plt.plot(test_dates, actual_prices, label='Actual Price', color='blue', linewidth=1.5)
plt.plot(test_dates, pred_prices, label='Predicted Price', color='red', linestyle='--', linewidth=1.2)
plt.plot(test_dates, current_prices, label='Previous Close', color='green', alpha=0.6, linewidth=1)
plt.title('JNJ Stock Price Prediction (LSTM)\nRMSE: ${:.2f}, MAPE: {:.2f}%'.format(
    np.sqrt(mean_squared_error(actual_prices, pred_prices)),
    np.mean(np.abs((actual_prices - pred_prices)/actual_prices))*100))
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Plot 2: Cumulative Returns Comparison
plt.subplot(2, 1, 2)
cum_actual = np.cumprod(1 + actual_returns) - 1
cum_pred = np.cumprod(1 + pred_returns) - 1
plt.plot(test_dates, cum_actual, label='Actual Returns', color='blue')
plt.plot(test_dates, cum_pred, label='Predicted Returns', color='red')
plt.title('Cumulative Returns Comparison')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "lstm_price_predictions.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {plots_dir}")
