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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================
# 1. LOAD RAW DATASETS (using your original paths)
# =====================
base_path = Path(__file__).resolve().parents[1]
raw_data_dir = base_path / "data" / "raw"
processed_data_dir = base_path / "data" / "processed"

# Load datasets (same as original)
dsf_jnj = pd.read_csv(raw_data_dir / "dsf_jnj.csv", parse_dates=['date'])
msenames = pd.read_csv(raw_data_dir / "msenames.csv", parse_dates=['NAMEDT'])

# =====================
# 2. DATA PREPARATION (Simplified)
# =====================
# Filter for PERMNO = 22111 (JNJ)
dsf_jnj = dsf_jnj[dsf_jnj['PERMNO'] == 22111]
df = dsf_jnj.merge(msenames, on='PERMNO', how='left')
df = df[['date', 'RET', 'VOL', 'PRC']].sort_values('date').reset_index(drop=True)

# Handle missing values
df['RET'] = df['RET'].fillna(0)
df = df.dropna()

# =====================
# 3. CREATE TRAIN/VAL/TEST SPLITS (same as original)
# =====================
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train_data = df[:train_size].copy()
val_data = df[train_size:train_size+val_size].copy()
test_data = df[train_size+val_size:].copy()

# =====================
# 4. SIMPLIFIED FEATURE ENGINEERING
# =====================
def create_simple_features(data):
    data = data.copy()
    # Basic features only
    data['log_return'] = np.log(data['PRC'] / data['PRC'].shift(1))
    data['log_vol'] = np.log(data['VOL'] / data['VOL'].shift(1))
    data['target'] = data['log_return'].shift(-1)  # Next day's return
    return data.dropna()

train_data = create_simple_features(train_data)
val_data = create_simple_features(val_data)
test_data = create_simple_features(test_data)

# Only use these 2 features
feature_cols = ['log_return', 'log_vol']

# =====================
# 5. SCALING (same as original)
# =====================
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(train_data[feature_cols])
y_train_scaled = target_scaler.fit_transform(train_data[['target']])

X_val_scaled = feature_scaler.transform(val_data[feature_cols])
y_val_scaled = target_scaler.transform(val_data[['target']])
X_test_scaled = feature_scaler.transform(test_data[feature_cols])
y_test_scaled = target_scaler.transform(test_data[['target']])

# =====================
# 6. CREATE SEQUENCES (same as original)
# =====================
def create_sequences(X, y, seq_length=10):
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

# =====================
# 7. SIMPLIFIED TRANSFORMER MODEL
# =====================
class SimpleStockTransformer(nn.Module):
    def __init__(self, feature_dim=2, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(100, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)

# =====================
# 8. TRAINING WITH EARLY STOPPING
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleStockTransformer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
no_improve_epochs = 0

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

for epoch in range(50):  # Increased max epochs
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x.to(device))
        loss = criterion(outputs, batch_y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(device))
            val_loss += criterion(outputs, batch_y.to(device)).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# =====================
# 9. EVALUATION (same metrics as original)
# =====================
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()

pred_returns = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
actual_returns = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Calculate metrics
print(f"\nReturn Prediction Results:")
print(f"MSE: {mean_squared_error(actual_returns, pred_returns):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual_returns, pred_returns)):.6f}")
print(f"MAE: {mean_absolute_error(actual_returns, pred_returns):.6f}")
print(f"R²: {r2_score(actual_returns, pred_returns):.4f}")
print(f"Directional Accuracy: {np.mean(np.sign(actual_returns) == np.sign(pred_returns)) * 100:.2f}%")

# Price prediction metrics
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
# 10. PRICE PREDICTION PLOTS
# =====================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create results directory if it doesn't exist
results_dir = base_path / "results"
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Get test dates and prices
test_dates = test_data['date'].iloc[10:10+len(pred_returns)].values  # Skip first 10 for sequence
current_prices = test_data['PRC'].iloc[10-1:10-1+len(pred_returns)].values
predicted_prices = current_prices * np.exp(pred_returns)
actual_prices = test_data['PRC'].iloc[10:10+len(pred_returns)].values

# Create figure with subplots
plt.figure(figsize=(15, 10))

# Plot 1: Price predictions vs actual
plt.subplot(2, 1, 1)
plt.plot(test_dates, actual_prices, label='Actual Price', color='blue', linewidth=1.5)
plt.plot(test_dates, predicted_prices, label='Predicted Price', color='red', linestyle='--', linewidth=1.2)
plt.plot(test_dates, current_prices, label='Previous Close (Baseline)', color='green', alpha=0.6, linewidth=1)
plt.title(f'JNJ Price Predictions\n(RMSE: {np.sqrt(mean_squared_error(actual_prices, predicted_prices)):.2f}, MAPE: {np.mean(np.abs((actual_prices - predicted_prices)/actual_prices))*100:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Plot 2: Scatter plot of actual vs predicted prices
plt.subplot(2, 1, 2)
plt.scatter(actual_prices, predicted_prices, alpha=0.6, s=20)
plt.plot([min(actual_prices), max(actual_prices)], 
         [min(actual_prices), max(actual_prices)], 
         'r--', alpha=0.8)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "jnj_baseline.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPrice prediction plots saved to: {plots_dir / 'jnj_baseline.png'}")
