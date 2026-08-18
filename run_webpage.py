import base64
import http.server
import io
import socketserver
import threading
import webbrowser
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 1. LOAD DATASET & FEATURE ENGINEERING
# =========================================================================
file_path = (
    r"C:\Users\lenovo\Downloads\AIML\Mid\cleaned_50k_household_power.csv"
)

print("Loading dataset...")
df = pd.read_csv(file_path)

target_col = (
    "Global_active_power"
    if "Global_active_power" in df.columns
    else "Global_Active_Power"
)

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.dropna(subset=["Datetime", target_col])
df = df.sort_values("Datetime").set_index("Datetime")

df_15m = df[[target_col]].resample("15min").mean().dropna()

calculated_threshold = df_15m[target_col].quantile(0.75)
y_binary = (df_15m[target_col] >= calculated_threshold).astype(int)

df_features = pd.DataFrame(index=df_15m.index)

df_features["sin_hour"] = np.sin(2 * np.pi * df_15m.index.hour / 24.0)
df_features["cos_hour"] = np.cos(2 * np.pi * df_15m.index.hour / 24.0)
df_features["is_weekend"] = df_15m.index.dayofweek.isin([5, 6]).astype(int)

for lag in [1, 2, 3, 4, 8, 96]:
  df_features[f"lag_{lag}"] = df_15m[target_col].shift(lag)

df_features["diff_1"] = (
    df_15m[target_col].shift(1) - df_15m[target_col].shift(2)
)
df_features["rolling_mean_4"] = df_15m[target_col].shift(1).rolling(4).mean()
df_features["rolling_std_4"] = df_15m[target_col].shift(1).rolling(4).std()
df_features["lag_ratio"] = (df_features["lag_1"] + 1e-5) / (
    df_features["lag_4"] + 1e-5
)

df_features = df_features.dropna()
y_binary = y_binary.loc[df_features.index]

X = df_features

# =========================================================================
# 2. TRAIN-TEST SPLIT & SCALING
# =========================================================================
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y_binary.iloc[:train_size], y_binary.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================================
# 3. MULTI-SEED VALIDATION
# =========================================================================
seed_values = [10, 42, 100, 500, 1234, 2026]
seed_results = []
test_predictions = None

for seed in seed_values:
  model = LogisticRegression(
      C=1.0,
      max_iter=2000,
      solver="lbfgs",
      class_weight="balanced",
      random_state=seed,
  )
  model.fit(X_train_scaled, y_train)

  y_pred = model.predict(X_test_scaled)
  test_predictions = y_pred

  acc = accuracy_score(y_test, y_pred) * 100
  prec = precision_score(y_test, y_pred, zero_division=0) * 100
  rec = recall_score(y_test, y_pred, zero_division=0) * 100
  f1 = f1_score(y_test, y_pred, zero_division=0) * 100

  seed_results.append({
      "Seed": seed,
      "Accuracy (%)": round(acc, 2),
      "Precision (%)": round(prec, 2),
      "Recall (%)": round(rec, 2),
      "F1-Score (%)": round(f1, 2),
  })

results_table = pd.DataFrame(seed_results)
final_accuracy = results_table["Accuracy (%)"].mean()

# =========================================================================
# 4. GENERATE MATPLOTLIB PLOT & CONVERT TO BASE64 IMAGE
# =========================================================================
test_results = pd.DataFrame(
    {"Actual": y_test, "Predicted": test_predictions}, index=y_test.index
)

last_ts = test_results.index.max()
start_ts = last_ts - pd.Timedelta(days=4)
df_plot = test_results.loc[test_results.index >= start_ts]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.step(
    df_plot.index,
    df_plot["Actual"],
    where="post",
    color="#1f77b4",
    linewidth=1.8,
    label="Actual Energy Load State",
)
ax1.set_title(
    f"Actual Load Category (75th Percentile Threshold = {calculated_threshold:.4f} kW)",
    fontsize=11,
    fontweight="bold",
)
ax1.set_ylabel("Load Category")
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Normal Load (0)", "Peak Load (1)"])
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right")

ax2.step(
    df_plot.index,
    df_plot["Predicted"],
    where="post",
    color="#d62728",
    linestyle="--",
    linewidth=1.8,
    label=(
        "Predicted Load State (Logistic Regression | Accuracy ="
        f" {final_accuracy:.2f}%)"
    ),
)
ax2.set_title(
    "Predicted Load Category (Standard model.predict)",
    fontsize=11,
    fontweight="bold",
)
ax2.set_xlabel("Datetime (15-Minute Intervals)", fontsize=10)
ax2.set_ylabel("Load Category")
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Normal Load (0)", "Peak Load (1)"])
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

plt.suptitle(
    "Actual vs Predicted Household Energy Load (75th Percentile Target)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()

# Save plot directly into memory buffer as PNG
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode("utf-8")
plt.close(fig)

# =========================================================================
# 5. GENERATE INDEX.HTML FILE
# =========================================================================
table_html = results_table.to_html(index=False, classes="dataframe")

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Household Power Prediction Dashboard</title>
    <style>
        body {{ font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        .metric {{ font-size: 1.2em; font-weight: bold; color: #27ae60; margin: 15px 0; }}
        table.dataframe {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        table.dataframe th, table.dataframe td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
        table.dataframe th {{ background-color: #3498db; color: white; }}
        table.dataframe tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Household Energy Load Peak Prediction</h1>
    
    <div class="card">
        <h2>Multi-Seed Validation Results</h2>
        <p><strong>75th Percentile Threshold:</strong> {calculated_threshold:.4f} kW</p>
        {table_html}
        <div class="metric">Mean Prediction Accuracy on Last 20% Test Data: {final_accuracy:.2f}%</div>
    </div>

    <div class="card">
        <h2>Actual vs Predicted Load State (Last 4 Days)</h2>
        <img src="data:image/png;base64,{img_base64}" alt="Energy Load Prediction Plot">
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
  f.write(html_content)

# =========================================================================
# 6. SERVE HTML ON LOCALHOST & OPEN BROWSER AUTOMATICALLY
# =========================================================================
PORT = 8000


class LocalHandler(http.server.SimpleHTTPRequestHandler):

  def log_message(self, format, *args):
    pass  # Suppress console logging


def start_server():
  with socketserver.TCPServer(("", PORT), LocalHandler) as httpd:
    httpd.serve_forever()


server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

print(f"Opening web dashboard at http://localhost:{PORT}...")
webbrowser.open(f"http://localhost:{PORT}")

# Keep main script alive
try:
  server_thread.join()
except KeyboardInterrupt:
  print("\nServer shut down.")