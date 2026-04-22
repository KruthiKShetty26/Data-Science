import pickle
import numpy as np
import tensorflow as tf
import os

data_dir = r'C:\Users\Kruthi K Shetty\data-science\data'
models_dir = r'C:\Users\Kruthi K Shetty\data-science\models'

# Load label encoder
with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
print("Classes:", le.classes_)

# Load model
model = tf.keras.models.load_model(os.path.join(models_dir, 'lung_model.keras'))

# Load test data
with open(os.path.join(data_dir, 'test_set.pkl'), 'rb') as f:
    test_data = pickle.load(f)

X_test = test_data['X_test']
y_test = test_data['y_test']

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix logic (simple)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
