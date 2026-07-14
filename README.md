# Customer Churn Prediction Project

Predicts customer churn risk from account, service, and engagement features.
Includes data generation, EDA, model training, and a deployable app.

## Project Structure
```
churn-project/
├── data/
│   └── customer_churn.csv          # synthetic dataset (5,000 rows)
├── docs/
│   ├── churn_analysis_report.md    # findings & recommendations
│   ├── chart_segments.png
│   └── chart_importance.png
├── generate_data.py                # reproducible data generation
├── analysis.py                     # EDA + model training + evaluation
├── model_export.json               # trained logistic regression weights
├── app.py                          # Streamlit predictor app (deploy this)
├── churn_risk_console.jsx          # React version of the same app
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Deploy for Free (Streamlit Community Cloud) — easiest option
1. Create a new GitHub repo, e.g. `churn-risk-console`.
2. Push these files to it (`app.py`, `model_export.json`, `requirements.txt` are the
   minimum needed — you can drop the other scripts in a `notebooks/` folder for reference).
   ```bash
   git init
   git add .
   git commit -m "Churn risk console"
   git branch -M main
   git remote add origin https://github.com/<your-username>/churn-risk-console.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io → sign in with GitHub → "New app".
4. Pick your repo, branch `main`, main file `app.py` → Deploy.
5. You get a free public URL like `churn-risk-console.streamlit.app` in ~2 minutes.

## Alternative: Deploy the React version
The `.jsx` file is a standalone component — to deploy it as a real site:
1. `npm create vite@latest churn-app -- --template react`
2. Replace `src/App.jsx` with `churn_risk_console.jsx`, `npm install lucide-react`.
3. `npm run build`, then push to GitHub.
4. Import the repo on https://vercel.com or https://netlify.com → auto-deploys on every push.

## Alternative: Hugging Face Spaces (also free, ML-community friendly)
1. Create a Space at https://huggingface.co/spaces → SDK: Streamlit.
2. Upload `app.py`, `model_export.json`, `requirements.txt` directly in the web UI, or
   push via git the same way as GitHub.
3. Space builds automatically and gives you a public URL.

## Retraining on Real Data
Replace `data/customer_churn.csv` with your real customer export (keep the same
column names, or update `analysis.py`'s `feature_cols`), then re-run:
```bash
python analysis.py
```
This regenerates `model_export.json` with new weights — `app.py` picks them up
automatically on next run/deploy.
