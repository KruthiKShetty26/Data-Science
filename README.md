# PneumoTrack: AI-Driven Clinical Respiratory Dashboard

PneumoTrack is a comprehensive clinical decision support system designed to assist pulmonologists in diagnosing and monitoring respiratory conditions through acoustic analysis of lung sounds. By combining Deep Learning, Temporal Analysis, and Explainable AI (XAI), PneumoTrack provides more than just a diagnosis—it provides actionable clinical insights.

## 🏥 Key Features
- **Real-Time Sound Classification**: Classifies breath cycles into Normal, Crackle, Wheeze, or Both.
- **Longitudinal Temporal Analysis**: Tracks acoustic biomarkers over time to calculate a **Clinical Risk Score (Stage 1-3)**.
- **Explainable AI (SHAP)**: Uses SHapley Additive exPlanations to visualize exactly which acoustic features influenced the AI's decision.
- **Acoustic Benchmarking**: Visual Mel-Spectrogram comparisons against healthy clinical norms.
- **Automated Clinical Reporting**: Generates a one-click PDF-ready report for patient records.

---

## 🔬 The Data Science Pipeline

The project is structured into 5 specialized Jupyter Notebooks:

1. **01_data_loading.ipynb**: Ingests raw ICBHI 2017 dataset annotations and consolidates patient metadata.
2. **02_preprocessing.ipynb**: Applies a **4th-order Butterworth Bandpass Filter** (100Hz–2000Hz) and standardizes audio to 3-second 22,050Hz segments.
3. **03_feature_extraction.ipynb**: Extracts a **76-dimensional acoustic fingerprint** including 26 MFCCs, 40 Mel-Spectrogram bands, Spectral Centroid, ZCR, and RMS Energy.
4. **04_temporal_model.ipynb**: The AI Engine. Uses **SMOTE** for class balancing and trains a **Multi-Layer Perceptron (MLP)** Neural Network.
5. **05_shap_explainability.ipynb**: Generates Global and Local explainability plots using SHAP to ensure diagnostic transparency.

---

## 🧠 Model Technical Details
- **Architecture**: 3-Layer Dense Neural Network (128 -> 64 -> 4 neurons).
- **Optimizer**: Adam (Adaptive Moment Estimation).
- **Generalization**: Dropout (0.3), EarlyStopping (patience 10), and StandardScaler normalization.
- **Performance**: High recall for clinical abnormalities (Crackles/Wheezes) with robust cross-validation.

---

## 📊 Understanding the Analytics

### 🔴 SHAP Waterfall Plots
- **Red Bars (+)**: Features that pushed the AI **toward** the diagnosis (e.g., high tonal energy for a Wheeze).
- **Blue Bars (-)**: Features that contradicted the diagnosis.
- Provides "Clinical Trust" by opening the AI black box.

### 📈 Temporal Trend Charts
- Tracks the **Slope of Deterioration**.
- An upward trend in RMS (Volume) or Spectral Centroid (Brightness) signals potential worsening of the patient's condition.

---

## 🛠️ Tech Stack
- **Languages**: Python (Flask, TensorFlow, Keras)
- **Signal Processing**: Librosa, SciPy
- **Explainability**: SHAP
- **Dashboard**: HTML5, CSS3, JavaScript (Plotly.js)
- **Version Control**: Git / GitHub

---

## 🚀 Installation & Usage
1. Clone the repository: `git clone https://github.com/KruthiKShetty26/Data-Science.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `python dashboard/app.py`
4. Access via browser: `http://127.0.0.1:5000`

---

