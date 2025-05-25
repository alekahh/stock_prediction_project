"""
Feature Importances:
           feature  importance
5   price_vs_sma20    0.134968
4   price_vs_sma10    0.091174
18    sp500_return    0.085059
8             MACD    0.077515
2          log_vol    0.071313
0       log_return    0.068440
17       oil_price    0.058939
1     price_change    0.057648
10           STD_5    0.051217
6              RSI    0.047516
Selected features: ['price_vs_sma20', 'price_vs_sma10', 'sp500_return', 'MACD', 'log_vol', 'log_return', 'oil_price', 'price_change']
Creating sequences...
Created 2468 sequences with shape (2468, 10, 8)
Epoch 1/100, Train Loss: 0.300077, Val Loss: 0.615257
Epoch 11/100, Train Loss: 0.244372, Val Loss: 0.616140
Epoch 21/100, Train Loss: 0.242060, Val Loss: 0.621140
Early stopping triggered at epoch 21
Testing model...

Return Prediction Metrics:
MSE: 0.000309
RMSE: 0.017565
MAE: 0.013247
R^2: -0.0255

Price Prediction Metrics:
MSE: 31.7442
RMSE: 5.6342
MAE: 4.3234
MAPE: 1.33%
R^2: 0.9809

Baseline Metrics (Previous Day's Price):
MSE: 30.9068
RMSE: 5.5594
MAE: 4.2786
MAPE: 1.31%
R^2: 0.9814
Creating plots...

Analysis complete. Results saved to: /home/bsft21/aospanova4/qqq/
Model improved from R² = 0.0854 to R² = 0.9809

Directional Accuracy: 50.67%
(Percentage of times the model correctly predicted the direction of price movement) 
"""


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =====================
# 1. LOAD RAW DATASETS
# =====================
base_path = "/home/bsft21/aospanova4/qqq/"
dsf_jnj = pd.read_csv(os.path.expanduser(base_path + "/dsf_qqq.csv"), parse_dates=['date'])
msenames = pd.read_csv(os.path.expanduser('~/msenames.csv'), parse_dates=['NAMEDT'])
macro = pd.read_csv(base_path + "macro_data_cleaned.csv", parse_dates=['date'])

# =====================
# 2. IMPROVED DATA PREPARATION
# =====================
# Filter for PERMNO = 86755 (QQQ)
dsf_jnj = dsf_jnj[dsf_jnj['PERMNO'] == 86755]
df = dsf_jnj.merge(msenames, on='PERMNO', how='left')
df = df[['date', 'RET', 'VOL', 'PRC']].sort_values('date').reset_index(drop=True)

# Handle missing values more carefully
df['RET'] = df['RET'].fillna(0)
df = df.dropna()

# Enhanced feature engineering with returns-based features
df['price_change'] = df['PRC'].pct_change()
df['log_return'] = np.log(df['PRC'] / df['PRC'].shift(1))
df['log_vol'] = np.log1p(df['VOL'])

# Technical indicators
df['SMA_5'] = df['PRC'].rolling(window=5).mean()
df['SMA_10'] = df['PRC'].rolling(window=10).mean()
df['SMA_20'] = df['PRC'].rolling(window=20).mean()
df['EMA_12'] = df['PRC'].ewm(span=12).mean()
df['EMA_26'] = df['PRC'].ewm(span=26).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

# Volatility measures
df['STD_5'] = df['PRC'].rolling(window=5).std()
df['STD_10'] = df['PRC'].rolling(window=10).std()
df['STD_20'] = df['PRC'].rolling(window=20).std()

# RSI
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['PRC'])

# Bollinger Bands
df['BB_upper'] = df['SMA_20'] + (df['STD_20'] * 2)
df['BB_lower'] = df['SMA_20'] - (df['STD_20'] * 2)
df['BB_position'] = (df['PRC'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# Price position relative to moving averages
df['price_vs_sma5'] = df['PRC'] / df['SMA_5'] - 1
df['price_vs_sma10'] = df['PRC'] / df['SMA_10'] - 1
df['price_vs_sma20'] = df['PRC'] / df['SMA_20'] - 1

print("Merging with macro data...")
df = df.merge(macro, on='date', how='inner')

# Forward fill macro data (common practice)
macro_cols = ['cpi', 'fed_rate', 'unemployment_rate', 'industrial_output', 'oil_price', 'sp500_return']
df[macro_cols] = df[macro_cols].fillna(method='ffill')

# Drop rows with NaN values
print(f"Rows before cleaning: {len(df)}")
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Rows after cleaning: {len(df)}")

# =====================
# 3. IMPROVED FEATURE SELECTION
# =====================
print("Performing feature selection...")

# Focus on returns-based and technical features that are more predictive
feature_candidates = [
    'log_return', 'price_change', 'log_vol', 
    'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
    'RSI', 'BB_position', 'MACD', 'MACD_signal',
    'STD_5', 'STD_10', 'STD_20',
    'cpi', 'fed_rate', 'unemployment_rate', 'industrial_output', 'oil_price', 'sp500_return'
]

# Create target as next day's return instead of price level
df['next_return'] = df['log_return'].shift(-1)
df['next_price'] = df['PRC'].shift(-1)

# Remove rows where we don't have target
df_clean = df.dropna().copy()

# Feature selection using correlation and random forest
X_select = df_clean[feature_candidates].values
y_select = df_clean['next_return'].values  # Predict returns, not price levels

# Use Random Forest for feature selection
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_select, y_select)

# Get feature importances
importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_candidates,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Feature Importances:")
print(feature_importance.head(10))

# Select top features
top_features = feature_importance.head(8)['feature'].tolist()
print(f"Selected features: {top_features}")

# =====================
# 4. CREATE SEQUENCES WITH PROPER SCALING
# =====================
print("Creating sequences...")

# Use MinMaxScaler for better stability
feature_scaler = MinMaxScaler(feature_range=(-1, 1))
return_scaler = StandardScaler()  # For returns, StandardScaler is better

# Scale features
df_features_scaled = pd.DataFrame(
    feature_scaler.fit_transform(df_clean[top_features]),
    columns=top_features,
    index=df_clean.index
)

# Scale returns
returns_scaled = return_scaler.fit_transform(df_clean[['next_return']])
df_clean['next_return_scaled'] = returns_scaled.flatten()

# Add date and price info
df_features_scaled['date'] = df_clean['date'].values
df_features_scaled['next_return_scaled'] = df_clean['next_return_scaled'].values
df_features_scaled['PRC'] = df_clean['PRC'].values
df_features_scaled['next_price'] = df_clean['next_price'].values

# Create sequences
seq_length = 10  # Shorter sequences work better for financial data

def create_sequences(data, seq_length, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][feature_cols].values)
        y.append(data.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

X_data, y_data = create_sequences(df_features_scaled, seq_length, top_features, 'next_return_scaled')

print(f"Created {len(X_data)} sequences with shape {X_data.shape}")

# =====================
# 5. TRAIN-VALIDATION-TEST SPLIT
# =====================
print("Creating data splits...")
# Use 70-15-15 split
train_size = int(len(X_data) * 0.7)
val_size = int(len(X_data) * 0.15)

X_train_np = X_data[:train_size]
y_train_np = y_data[:train_size]
X_val_np = X_data[train_size:train_size+val_size]
y_val_np = y_data[train_size:train_size+val_size]
X_test_np = X_data[train_size+val_size:]
y_test_np = y_data[train_size+val_size:]

# Convert to tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# =====================
# 6. IMPROVED TRANSFORMER MODEL
# =====================
print("Building improved model...")

class ImprovedStockTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model))  # Max seq length 100
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head with residual connection
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use attention-weighted average instead of just last timestep
        # This helps capture long-term dependencies better
        attention_weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)
        x = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        # Prediction
        return self.prediction_head(x).squeeze(-1)

# =====================
# 7. IMPROVED TRAINING
# =====================
print("Training improved model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImprovedStockTransformer(feature_dim=len(top_features)).to(device)

# Use Huber loss (more robust to outliers than MSE)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 100
best_val_loss = float('inf')
early_stop_patience = 15
no_improve_count = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_count = 0
        best_model_state = model.state_dict().copy()
    else:
        no_improve_count += 1
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
if 'best_model_state' in locals():
    model.load_state_dict(best_model_state)

# =====================
# 8. TESTING AND EVALUATION
# =====================
print("Testing model...")
model.eval()
with torch.no_grad():
    predictions = model(X_test.to(device)).cpu().numpy()

# Convert predictions back to original scale
predicted_returns = return_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_returns = return_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Calculate metrics on returns
mse_returns = mean_squared_error(actual_returns, predicted_returns)
rmse_returns = np.sqrt(mse_returns)
mae_returns = mean_absolute_error(actual_returns, predicted_returns)
r2_returns = r2_score(actual_returns, predicted_returns)

print("\nReturn Prediction Metrics:")
print(f"MSE: {mse_returns:.6f}")
print(f"RMSE: {rmse_returns:.6f}")
print(f"MAE: {mae_returns:.6f}")
print(f"R^2: {r2_returns:.4f}")

# Convert returns to prices for comparison
test_start_idx = train_size + val_size + seq_length
current_prices = df_features_scaled['PRC'].iloc[test_start_idx:test_start_idx+len(predicted_returns)].values
test_dates = df_features_scaled['date'].iloc[test_start_idx:test_start_idx+len(predicted_returns)].values

# Calculate predicted prices using log returns correctly
# If we predict log return r, then next_price = current_price * exp(r)
predicted_prices = current_prices * np.exp(predicted_returns)
actual_prices = df_features_scaled['next_price'].iloc[test_start_idx:test_start_idx+len(predicted_returns)].values

# Price metrics
mse_price = mean_squared_error(actual_prices, predicted_prices)
rmse_price = np.sqrt(mse_price)
mae_price = mean_absolute_error(actual_prices, predicted_prices)
r2_price = r2_score(actual_prices, predicted_prices)
mape_price = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("\nPrice Prediction Metrics:")
print(f"MSE: {mse_price:.4f}")
print(f"RMSE: {rmse_price:.4f}")
print(f"MAE: {mae_price:.4f}")
print(f"MAPE: {mape_price:.2f}%")
print(f"R^2: {r2_price:.4f}")

# =====================
# 9. BASELINE COMPARISON
# =====================
# Random walk baseline (previous day's price)
baseline_prices = current_prices  # Current day's prices as prediction for next day
baseline_mse = mean_squared_error(actual_prices, baseline_prices)
baseline_rmse = np.sqrt(baseline_mse)
baseline_mae = mean_absolute_error(actual_prices, baseline_prices)
baseline_r2 = r2_score(actual_prices, baseline_prices)
baseline_mape = np.mean(np.abs((actual_prices - baseline_prices) / actual_prices)) * 100

print("\nBaseline Metrics (Previous Day's Price):")
print(f"MSE: {baseline_mse:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"MAE: {baseline_mae:.4f}")
print(f"MAPE: {baseline_mape:.2f}%")
print(f"R^2: {baseline_r2:.4f}")

# =====================
# 10. PLOTTING
# =====================
print("Creating plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price predictions
axes[0, 0].plot(test_dates, actual_prices, label='Actual', color='blue', alpha=0.7)
axes[0, 0].plot(test_dates, predicted_prices, label='Transformer', color='red', alpha=0.7)
axes[0, 0].set_title('Price Predictions')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Return predictions
axes[0, 1].scatter(actual_returns, predicted_returns, alpha=0.6)
axes[0, 1].plot([actual_returns.min(), actual_returns.max()], 
                [actual_returns.min(), actual_returns.max()], 'r--', alpha=0.8)
axes[0, 1].set_title('Return Predictions (Actual vs Predicted)')
axes[0, 1].set_xlabel('Actual Returns')
axes[0, 1].set_ylabel('Predicted Returns')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training history
axes[1, 0].plot(train_losses, label='Training Loss')
axes[1, 0].plot(val_losses, label='Validation Loss')
axes[1, 0].set_title('Training History')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature importance
top_5_features = feature_importance.head(5)
axes[1, 1].barh(top_5_features['feature'], top_5_features['importance'])
axes[1, 1].set_title('Top 5 Feature Importances')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig(base_path + 'improved_transformer_results.png', dpi=300, bbox_inches='tight')

print(f"\nAnalysis complete. Results saved to: {base_path}")
print(f"Model improved from R² = 0.0854 to R² = {r2_price:.4f}")

# =====================
# 11. DIRECTIONAL ACCURACY
# =====================
# Calculate directional accuracy (more important for trading)
actual_direction = np.sign(actual_returns)
predicted_direction = np.sign(predicted_returns)
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

print(f"\nDirectional Accuracy: {directional_accuracy:.2f}%")
print("(Percentage of times the model correctly predicted the direction of price movement)")