import pandas as pd
import joblib
import os

data_path = r'C:\Users\Kruthi K Shetty\data-science\data\features_df.csv'
models_dir = r'C:\Users\Kruthi K Shetty\data-science\models'

df = pd.read_csv(data_path)
meta_cols = ['npy_index', 'patient_id', 'filename', 'cycle_start', 'cycle_end', 'label']
feature_cols = [c for c in df.columns if c not in meta_cols]

# Save it
joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.pkl'))
print(f"Created feature_cols.pkl with {len(feature_cols)} features.")
