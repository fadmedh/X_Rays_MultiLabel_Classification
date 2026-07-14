# CheXpert Chest X-Ray Diagnostic Assistant on Streamlit Cloud

A professional, clinical-grade Streamlit web application that deploys a fine-tuned DenseNet-121 model trained on the CheXpert dataset. The tool performs multi-label classification across 10 pathological target classes.

## Features
- **Modern Medical UI**: Standard columns, clinical color palettes, custom responsive CSS card layout, and medical status panel.
- **Detailed Analytics**: Interactive Plotly horizontal bar chart sorted by confidence.
- **Custom Clinical Indicators**: Dynamic, colored progress bars indicating probability levels (low, moderate, high risk).
- **Automated Alerts**: Real-time high-risk findings alerts and screening diagnostic summaries.
- **Full Python 3.13 Compatibility**: Optimized configuration using TensorFlow 2.18+ / Keras 3.

---

## Repository Structure
Ensure your files are organized as follows:
```text
MultiLabel_Classification/
├── .streamlit/
│   └── config.toml               # Custom Streamlit theme (primary medical colors)
├── app.py                        # Main Streamlit dashboard interface
├── utils.py                      # Preprocessing & cached model loading functions
├── best_model_finetuned.keras    # Pre-trained Keras model (or best_model.h5)
├── requirements.txt              # Strict Python 3.13 compatible dependencies
└── README.md                     # Documentation
```

---

## Local Setup under Python 3.13

### Prerequisites
Make sure you have **Python 3.13** installed on your system.

### 1. Clone & Navigate to Repository
```bash
git clone <repository-url>
cd MultiLabel_Classification
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the lightweight, Python 3.13 native versions of the required libraries:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
This will start the development server and automatically open the application in your default web browser (typically at `http://localhost:8501`).

---

## Deployment to Streamlit Cloud

To deploy your medical analyzer on Streamlit Cloud, follow these steps:

1. **Commit and Push to GitHub**:
   Ensure all 5 app files (`app.py`, `utils.py`, `requirements.txt`, `.streamlit/config.toml`, and `README.md`) along with your pre-trained model file (`best_model_finetuned.keras` or `best_model.h5`) are committed and pushed to a public or private GitHub repository.
   
   *Note: If your model file is too large for GitHub (over 100MB), configure Git Large File Storage (LFS) or load the model directly from a cloud storage link in `utils.py`.*

2. **Deploy on Streamlit Community Cloud**:
   - Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your Repository, Branch, and Main file path (`app.py`).
   - Under **Advanced settings...**, ensure that Python Version is set to **3.13** (if selected automatically or select standard default).
   - Click **"Deploy!"**. Streamlit Cloud will read `requirements.txt`, install the pre-compiled Python 3.13 wheels, apply the theme from `.streamlit/config.toml`, and launch your app.

---

## Model Outputs
The system classifies chest X-rays across 10 target classes:
1. `Enlarged Cardiomediastinum`
2. `Cardiomegaly`
3. `Lung Opacity`
4. `Edema`
5. `Consolidation`
6. `Pneumonia`
7. `Atelectasis`
8. `Pneumothorax`
9. `Pleural Effusion`
10. `Support Devices`

---

## Disclaimer
*This tool is intended for screening and educational assistance only. It should not be used as a primary diagnostic instrument. All findings must be clinically correlated and verified by a board-certified radiologist.*
