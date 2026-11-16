# 🔍 CrimeLens — Terminal Commands Reference

This file contains all the terminal commands needed to run, test, and demonstrate the project on **localhost**. You can copy and paste these commands directly into your terminal.

---

## 🚀 Quick Start (One-Command Launch)

### Option A: Launch Interactive Streamlit Frontend (Recommended for Demonstration)
Open **PowerShell** or **Command Prompt** in this project folder:

```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
> Once running, open your browser to: **http://localhost:8501**  
> *(Press `Ctrl + C` in the terminal to stop the server)*

---

### Option B: Launch Jupyter Notebook
To open and inspect the full academic machine learning implementation:

```powershell
.venv\Scripts\jupyter.exe notebook Crime_Detection_NLP.ipynb
```
> Once running, open your browser to: **http://localhost:8888**  
> *(All 13 sections, charts, tables, and models are already pre-computed and saved)*

---

## 📋 Complete Step-by-Step Setup Commands

If you ever need to set up the environment from scratch on a new machine, run these commands in order:

### 1. Navigate to Project Directory
```powershell
cd "c:\Users\sharm\OneDrive\Desktop\NLP Project"
```

---

### 2. Activate the Virtual Environment

**In PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run the activate command)*

**In Command Prompt (cmd.exe):**
```cmd
.venv\Scripts\activate.bat
```

---

### 3. Install Required Dependencies
*(Only needed if setting up on a fresh machine; packages are already installed in `.venv`)*
```powershell
pip install -r requirements.txt
```

---

### 4. Download Required NLTK Corpora (Offline Text Processing)
*(Already downloaded in your environment)*
```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

---

### 5. Re-Execute the Entire Notebook Headlessly (Optional)
If you want to re-run all 13 notebook sections from start to finish via terminal:
```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace Crime_Detection_NLP.ipynb
```

---

## 🛠️ Troubleshooting & Port Management

### Issue: "Port 8501 is not available"
If Streamlit is already running in another terminal window and blocking port 8501:

**In PowerShell (kill existing Streamlit process):**
```powershell
Get-Process -Name streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Alternative: Run on a Different Port:**
```powershell
.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8502
```
Then visit: **http://localhost:8502**

---

## 📌 Summary of Key Files

| File / Folder | Purpose |
| :--- | :--- |
| `app/streamlit_app.py` | Modern web frontend (5 pages: Home, Prediction, Performance, Dataset, About) |
| `Crime_Detection_NLP.ipynb` | Complete academic Jupyter Notebook with all 13 sections executed |
| `models/crime_classifier.joblib` | Trained, serialized production pipeline (TF-IDF + Naive Bayes) |
| `models/model_metadata.json` | Model hyperparameters, training date, and empirical test metrics |
| `data/raw/CrimeVsNoCrimeArticles.csv` | Preserved original Kaggle dataset (7,124 records) |
| `data/processed/CrimeVsNoCrimeArticles_cleaned.csv` | Cleaned, deduplicated dataset (7,116 records) |
| `README.md` | In-depth academic documentation and methodology report |

