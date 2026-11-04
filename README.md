# Information Integrity Monitor — ML prototype

## Files
- `app.py`: Streamlit dashboard with dark theme and ML output panel.
- `train_model.py`: trains a TF-IDF + Logistic Regression review-priority model.
- `training_data.csv`: small synthetic training examples.
- `sample_data.csv`: fictional claims displayed in the dashboard.
- `requirements.txt`: Python packages.

## Windows setup (PowerShell)
Open PowerShell in this folder and activate your existing virtual environment if you have one:
```powershell
.venv\Scripts\Activate.ps1
```

Install packages:
```powershell
python -m pip install -r requirements.txt
```

Train the demonstration model:
```powershell
python train_model.py
```

Start the dashboard:
```powershell
python -m streamlit run app.py
```

If Streamlit is already running, stop it in the terminal with Ctrl+C before restarting.

## Important limitations
This is a research/classroom prototype using synthetic data. The model predicts a review-priority label, not truth or falsehood. Its test score is only a result on a tiny synthetic split and must not be presented as real-world model accuracy. Do not use private communications or sensitive operational data. Any real-world response must be reviewed and approved by an authorised human.
