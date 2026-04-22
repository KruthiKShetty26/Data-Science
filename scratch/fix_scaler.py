import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Paths
data_dir = r'C:\Users\Kruthi K Shetty\data-science\data'
models_dir = r'C:\Users\Kruthi K Shetty\data-science\models'
os.makedirs(models_dir, exist_ok=True)

print("Loading features...")
df_path = os.path.join(data_dir, 'features_df.csv')
df = pd.read_csv(df_path)

# Meta columns to exclude from features
meta_cols = ['npy_index', 'patient_id', 'filename', 'cycle_start', 'cycle_end', 'label', 'diagnosis']
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].to_numpy()
y = df['label'].to_numpy()

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Fitting scaler on {len(feature_cols)} features...")
# We use the same split as the training notebook (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Scaler saved to {scaler_path}")

# Update test_set.pkl as well
test_set_path = os.path.join(data_dir, 'test_set.pkl')
with open(test_set_path, 'wb') as f:
    pickle.dump({'X_test': scaler.transform(X_test), 'y_test': y_test, 'scaler': scaler}, f)

print(f"Updated {test_set_path} with the fitted scaler.")
