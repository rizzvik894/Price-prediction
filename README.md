# XRP Next-Day Price Prediction (Random Forest)

Mini project for **CCE3131 AI/ML Lab**, Department of Computer and Communication Engineering, Manipal University Jaipur.
By Ritvik Chauhan.

The project predicts the next-day closing price of XRP/USDT with a **Random Forest regressor**, tunes it with
time-ordered cross-validation, tests it day by day on the last 90 days and compares it with a naive
"tomorrow equals today" baseline.

## What is in this repository

| File | What it is |
|---|---|
| `app.py` | Streamlit app (forecast, test results, feature importance) |
| `requirements.txt` | Python packages the app needs |
| `XRPUSDT20182024new.csv` | Daily XRP/USDT prices, 26 May 2018 to 26 May 2024 |
| `XRP_RandomForest_Notebook.ipynb` | Google Colab notebook with the full project (cleaning, features, tuning, testing, report) |

## How the model works

1. The CSV is cleaned and put on a continuous daily calendar.
2. Prices are turned into daily returns, and 9 features are built from the past only
   (5 lagged returns, 7- and 14-day average return, 7- and 14-day volatility).
3. A Random Forest predicts the next-day return, which is turned into a price.
4. The data is split by time: the last 90 days are the test set. Each test day is predicted one day ahead
   using only earlier days, and the forest is refitted every 5 days.
5. Hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`) are chosen with 5-fold time-series
   cross-validation. The best settings were 100 trees, `max_depth` = 2 and `min_samples_leaf` = 50.

**Result:** the tuned forest only matches the naive baseline, and its up/down calls are close to chance,
because past returns carry very little information about the next day.

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `XRPUSDT20182024new.csv` from the same folder, or you can upload the file in the sidebar.

## Deploy on Streamlit Community Cloud (free)

1. Push these files to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repository, set the main file to `app.py` and deploy.

## Run the notebook

Open `XRP_RandomForest_Notebook.ipynb` in Google Colab, choose **Runtime > Run all** and upload
`XRPUSDT20182024new.csv` when the first cell asks for it.

## Data

Kaggle dataset **XRPUSDT20182024new** by kryusufkaya:
https://www.kaggle.com/datasets/kryusufkaya/xrpusdt20182024new

## Disclaimer

This is a lab exercise and not financial advice.
