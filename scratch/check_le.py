import pickle
import os

models_dir = r'C:\Users\Kruthi K Shetty\data-science\models'
le_path = os.path.join(models_dir, 'label_encoder.pkl')

if os.path.exists(le_path):
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    print(f"Label Encoder Classes: {le.classes_}")
    for i, cls in enumerate(le.classes_):
        print(f"{i} -> {cls}")
else:
    print("Label encoder not found.")
