"""XRP next-day price prediction with a Random Forest (basic Streamlit app).

Run:  streamlit run app.py
Needs XRPUSDT20182024new.csv in the same folder (or upload it in the sidebar).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV_NAME = "XRPUSDT20182024new.csv"
FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag5", "rm7", "rm14", "rs7", "rs14"]

st.set_page_config(page_title="XRP Price Prediction", layout="wide")
st.title("XRP Next-Day Price Prediction")
st.caption("Random Forest regression | CCE3131 AI/ML Lab mini project | Ritvik Chauhan")


# ------------------------------------------------------------------ data
@st.cache_data
def load_data(file):
    raw = pd.read_csv(file)
    df = raw.rename(columns={"Price": "Close", "Vol": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")
    df = df.asfreq("D").ffill()                       # one row per calendar day
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data
def make_features(close):
    ret = np.log(close).diff().dropna()               # daily log-return
    feat = pd.DataFrame({f"lag{i}": ret.shift(i - 1) for i in range(1, 6)})
    feat["rm7"], feat["rm14"] = ret.rolling(7).mean(), ret.rolling(14).mean()
    feat["rs7"], feat["rs14"] = ret.rolling(7).std(), ret.rolling(14).std()
    data = pd.concat([feat, (ret.shift(-1) * 100).rename("target")], axis=1).dropna()   # target: next-day return in %
    return feat, data


def forest(n_estimators, max_depth, min_samples_leaf):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)


@st.cache_data(show_spinner=False)
def walk_forward(close, test_days, n_estimators, max_depth, min_samples_leaf):
    """Predict each test day one day ahead; the forest is refitted every 5 days."""
    feat, data = make_features(close)
    y = close.values.astype(float)
    n_obs = len(y); train_n = n_obs - test_days
    preds, mdl = [], None
    for k, t in enumerate(range(train_n - 1, n_obs - 1)):
        i = data.index.get_loc(close.index[t])
        if mdl is None or k % 5 == 0:
            mdl = forest(n_estimators, max_depth, min_samples_leaf).fit(data.iloc[:i][FEATURES], data.iloc[:i]["target"])
        preds.append(y[t] * np.exp(mdl.predict(data.iloc[[i]][FEATURES])[0] / 100))
    return np.array(preds)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload the XRP CSV (optional)", type="csv")
test_days = st.sidebar.slider("Test period (days)", 30, 180, 90, step=10)
st.sidebar.subheader("Random Forest settings")
n_estimators = st.sidebar.slider("n_estimators (trees)", 50, 300, 100, step=50)
max_depth = st.sidebar.slider("max_depth", 1, 12, 2)
min_leaf = st.sidebar.slider("min_samples_leaf", 1, 100, 50)
st.sidebar.caption("Defaults (100, 2, 50) are the best settings found by time-series cross-validation in the notebook.")

source = uploaded if uploaded is not None else (CSV_NAME if os.path.exists(CSV_NAME) else None)
if source is None:
    st.warning(f"Put {CSV_NAME} in the same folder as app.py, or upload it using the sidebar.")
    st.stop()

df = load_data(source)
close = df["Close"]
feat, data = make_features(close)
n_obs = len(close); train_n = n_obs - test_days
y = close.values.astype(float)

with st.spinner("Training and testing the forest..."):
    preds = walk_forward(close, test_days, n_estimators, max_depth, min_leaf)
actual, prev = y[train_n:], y[train_n - 1:-1]
dates = close.index[train_n:]
naive = prev


def scores(p):
    mae, mse = mean_absolute_error(actual, p), mean_squared_error(actual, p)
    return {"MAE (USD)": mae, "MSE (USD^2)": mse, "RMSE (USD)": np.sqrt(mse),
            "MAPE (%)": np.mean(np.abs((actual - p) / actual)) * 100}


rf_s, nv_s = scores(preds), scores(naive)
dir_acc = np.mean(np.sign(preds - prev) == np.sign(actual - prev)) * 100
wins = int(np.sum(np.abs(actual - preds) < np.abs(actual - naive)))

# next-day forecast: forest refitted on all data
final = forest(n_estimators, max_depth, min_leaf).fit(data[FEATURES], data["target"])
p_next = float(y[-1] * np.exp(final.predict(feat.dropna().iloc[[-1]][FEATURES])[0] / 100))
rel = actual / preds - 1
lo, hi = p_next * (1 + np.percentile(rel, 5)), p_next * (1 + np.percentile(rel, 95))
next_date = close.index[-1] + pd.Timedelta(days=1)

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(["Next-day forecast", "Test results", "Feature importance", "About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast for {next_date.date()}", f"{p_next:.4f} USD", f"{p_next - y[-1]:+.4f} vs last close")
    c2.metric("Last close in the file", f"{y[-1]:.4f} USD", str(close.index[-1].date()))
    c3.metric("90% interval", f"{lo:.4f} to {hi:.4f}")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(close.index[-60:], close.iloc[-60:], color="#1f4e79", label="Last 60 days")
    ax.errorbar([next_date], [p_next], yerr=[[p_next - lo], [hi - p_next]], fmt="o", color="#c55a11", capsize=6, label="Forecast (90% interval)")
    ax.set_ylabel("USD"); ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.caption("The interval comes from the 5th and 95th percentile of the forest's percentage errors on the test days. It is rough.")

with tab2:
    st.subheader(f"Last {test_days} days, predicted one day ahead")
    table = pd.DataFrame({"Random Forest": rf_s, "Naive (tomorrow = today)": nv_s}).T
    st.dataframe(table.style.format({"MAE (USD)": "{:.4f}", "MSE (USD^2)": "{:.6f}", "RMSE (USD)": "{:.4f}", "MAPE (%)": "{:.2f}"}))
    c1, c2 = st.columns(2)
    c1.metric("Days the forest beat the naive forecast", f"{wins} of {test_days}")
    c2.metric("Up/down direction right", f"{dir_acc:.1f}%", help="About 50% is the same as guessing.")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dates, actual, color="black", lw=2, label="Actual")
    ax.plot(dates, naive, color="#7f7f7f", lw=1, label="Naive")
    ax.plot(dates, preds, color="#c55a11", lw=1.4, label="Random Forest")
    ax.set_ylabel("USD"); ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)
    day = st.selectbox("Pick a test day", [str(d.date()) for d in dates], index=len(dates) // 2)
    j = [str(d.date()) for d in dates].index(day)
    st.write(f"**{day}**: actual close {actual[j]:.4f} USD | forest predicted {preds[j]:.4f} | "
             f"naive predicted {naive[j]:.4f} | previous close {prev[j]:.4f}")

with tab3:
    imp = pd.Series(final.feature_importances_ * 100, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp.index, imp.values, color="#1f4e79"); ax.set_xlabel("Importance (%)"); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.write("lag1..lag5 = returns of today and the 4 days before; rm7/rm14 = average return over 7/14 days; "
             "rs7/rs14 = volatility over 7/14 days. Importance shows which features the forest used most, not whether they push the price up or down.")

with tab4:
    st.markdown("""
**What this app does.** It predicts the next-day closing price of XRP/USDT with a **Random Forest regressor**.

1. The CSV is cleaned and put on a daily calendar.
2. Prices are turned into daily returns, and 9 features are built from the past only.
3. The forest predicts tomorrow's return, which is turned into a price.
4. The test is done day by day: each test day is predicted using only earlier days.
5. The forest is compared with a naive baseline that says "tomorrow equals today".

**Result in the notebook:** the tuned forest only matches the naive baseline and its up/down calls are close to chance, because past returns carry very little information about the next day.

*This is a lab exercise, not financial advice.*
""")
