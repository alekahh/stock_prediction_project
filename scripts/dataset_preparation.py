import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Paths
base_path = Path(__file__).resolve().parents[1]
data_dir = base_path / "data"
raw_dir = data_dir / "raw"
macro_dir = data_dir / "processed"
macro_dir.mkdir(parents=True, exist_ok=True)


def load_and_merge_data():
    qqq = pd.read_csv(raw_dir / "TSM.csv")
    macro = pd.read_csv(macro_dir / "macro_data_cleaned.csv")

    if 'close' in macro.columns:
        macro = macro.drop(columns=['close'])

    qqq['Date'] = pd.to_datetime(qqq['Date']).dt.tz_localize(None)
    macro['Date'] = pd.to_datetime(macro['Date']).dt.tz_localize(None)

    qqq = qqq.sort_values('Date').reset_index(drop=True)
    macro = macro.sort_values('Date').reset_index(drop=True)

    merged = pd.merge_asof(
        qqq,
        macro,
        left_on='Date',
        right_on='Date',
        direction='backward'
    )

    macro_cols = ['cpi', 'fed_rate', 'unemployment_rate', 'industrial_output', 'oil_price']
    merged[macro_cols] = merged[macro_cols].fillna(method='ffill', limit=5)
    return merged


def create_alpha_features(df):
    df = df.copy()
    df['return_1d'] = df['Close'].shift(1).pct_change()
    df['return_5d'] = df['Close'].shift(1).pct_change(5)
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    df['sma_5'] = df['Close'].shift(1).rolling(5, min_periods=3).mean()
    df['sma_20'] = df['Close'].shift(1).rolling(20, min_periods=5).mean()
    df['price_to_sma5'] = (df['Close'] - df['sma_5']) / df['sma_5']
    df['price_to_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']

    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20, min_periods=5).mean()
    df['volume_change'] = df['Volume'].pct_change()

    df['volatility_5d'] = df['return_1d'].rolling(5, min_periods=3).std()
    df['volatility_20d'] = df['return_1d'].rolling(20, min_periods=5).std()

    if 'fed_rate' in df.columns:
        df['fed_rate_change'] = df['fed_rate'].diff()
    if 'unemployment_rate' in df.columns:
        df['unemployment_change'] = df['unemployment_rate'].diff()

    df['sentiment_3d_avg'] = df['Scaled_sentiment'].rolling(3, min_periods=2).mean()
    df['sentiment_5d_avg'] = df['Scaled_sentiment'].rolling(5, min_periods=3).mean()

    df['target'] = df['Close'].shift(-1) / df['Close'] - 1
    return df


def select_features_with_rf(df, importance_threshold=0.01):
    exclude_cols = ['Date', 'target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    split_date = df['Date'].quantile(0.8)
    train_data = df[df['Date'] <= split_date].copy()

    X_train = train_data[feature_cols].dropna()
    y_train = train_data.loc[X_train.index, 'target']
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[mask]
    y_train = y_train[mask]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    selected_features = importance_df[importance_df['importance'] >= importance_threshold]

    print(f"Selected {len(selected_features)} features above threshold {importance_threshold}")
    print(selected_features.head(15).to_string(index=False))

    importance_df.to_csv(macro_dir / "TSM_all_feature_importance.csv", index=False)
    return selected_features


def save_final_data(df, selected_features):
    feature_names = selected_features['feature'].tolist()

    # Keep Date, selected features, target, and Close for later use (e.g., plotting)
    final_cols = ['Date', 'Close'] + feature_names + ['target']
    final_df = df[final_cols].dropna()

    final_df.to_csv(macro_dir / "TSM_data_with_selected_features.csv", index=False)
    selected_features.to_csv(macro_dir / "TSM_selected_feature_importance.csv", index=False)

    print(f"Saved dataset with {len(feature_names)} selected features ({len(final_df)} samples), including 'Close'")



if __name__ == "__main__":
    merged_df = load_and_merge_data()
    featured_df = create_alpha_features(merged_df)
    selected_features = select_features_with_rf(featured_df, importance_threshold=0.01)
    save_final_data(featured_df, selected_features)


