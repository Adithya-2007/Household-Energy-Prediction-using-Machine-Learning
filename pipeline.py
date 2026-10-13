import os, json, webbrowser
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
)

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "full_household_power_15m.csv")
    raw_txt_path = os.path.join(base_dir, "..", "household_power_consumption.txt")

    # 1. Dataset Ingestion & 15-Minute Resampling
    if os.path.exists(csv_path):
        print("Loading full dataset from full_household_power_15m.csv...")
        df_15m = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    elif os.path.exists(raw_txt_path):
        print("Parsing raw UCI household_power_consumption.txt (2,075,259 records)...")
        df_raw = pd.read_csv(raw_txt_path, sep=";", na_values=["?"], low_memory=False)
        df_raw["Datetime"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df_raw = df_raw.dropna(subset=["Datetime", "Global_active_power"]).sort_values("Datetime").set_index("Datetime")
        cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        df_15m = df_raw[cols].resample("15min").mean().dropna()
        df_15m.to_csv(csv_path)
    else:
        raise FileNotFoundError("Could not locate full dataset file.")

    target_col = "Global_active_power"
    threshold_75 = float(df_15m[target_col].quantile(0.75))
    baseline_mean = float(df_15m[target_col].mean())
    y_reg_raw = df_15m[target_col]
    y_class_raw = (df_15m[target_col] >= threshold_75).astype(int)

    # 2. Feature Engineering (Temporal, Momentum, 24-hr Novelty Lags)
    print("Engineering temporal, momentum, and sub-metering features...")
    X = pd.DataFrame(index=df_15m.index)
    X["sin_hour"] = np.sin(2 * np.pi * df_15m.index.hour / 24.0)
    X["cos_hour"] = np.cos(2 * np.pi * df_15m.index.hour / 24.0)
    X["dayofweek"] = df_15m.index.dayofweek
    X["is_weekend"] = df_15m.index.dayofweek.isin([5, 6]).astype(int)

    for lag in [1, 2, 3, 4, 8, 96]:
        X[f"lag_{lag}"] = df_15m[target_col].shift(lag)

    X["diff_1"] = df_15m[target_col].shift(1) - df_15m[target_col].shift(2)
    X["rolling_mean_4"] = df_15m[target_col].shift(1).rolling(4).mean()
    X["rolling_std_4"] = df_15m[target_col].shift(1).rolling(4).std()
    X["rolling_mean_96"] = df_15m[target_col].shift(1).rolling(96).mean()
    X["rolling_std_96"] = df_15m[target_col].shift(1).rolling(96).std()

    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Voltage", "Global_reactive_power"]:
        if col in df_15m.columns:
            X[f"{col}_lag_1"] = df_15m[col].shift(1)
            X[f"{col}_lag_96"] = df_15m[col].shift(96)

    valid_idx = X.dropna().index
    X, y_reg, y_class = X.loc[valid_idx], y_reg_raw.loc[valid_idx], y_class_raw.loc[valid_idx]

    # 3. 80/20 Chronological Split & Standard Scaling
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_r_train, y_r_test = y_reg.iloc[:train_size], y_reg.iloc[train_size:]
    y_c_train, y_c_test = y_class.iloc[:train_size], y_class.iloc[train_size:]
    print(f"Dataset split: Train = {len(X_train):,} intervals (80%) | Test = {len(X_test):,} intervals (20%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training
    print("Training Tier 1: Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_c_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    print("Training Tier 2: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_r_train)
    rf_pred = rf.predict(X_test)

    print("Training Tier 3: ANN (MLPRegressor)...")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True, max_iter=300, random_state=42)
    mlp.fit(X_train_scaled, y_r_train)
    mlp_pred = mlp.predict(X_test_scaled)

    # 5. Benchmarking & Efficiency Evaluation
    cm = confusion_matrix(y_c_test, lr_pred)
    tn, fp, fn, tp = cm.ravel()
    tot = len(y_c_test)
    y_range = y_r_test.max() - y_r_test.min()

    lr_acc = accuracy_score(y_c_test, lr_pred) * 100
    lr_auc = roc_auc_score(y_c_test, lr_prob) * 100
    lr_prec = precision_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_rec = recall_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_spec = (tn / (tn + fp)) * 100
    lr_f1 = f1_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_mcc = matthews_corrcoef(y_c_test, lr_pred)
    lr_bal = balanced_accuracy_score(y_c_test, lr_pred) * 100

    rf_mae, rf_rmse = mean_absolute_error(y_r_test, rf_pred), root_mean_squared_error(y_r_test, rf_pred)
    rf_mse, rf_r2 = mean_squared_error(y_r_test, rf_pred), r2_score(y_r_test, rf_pred)
    rf_acc = (1 - rf_mae / y_range) * 100

    mlp_mae, mlp_rmse = mean_absolute_error(y_r_test, mlp_pred), root_mean_squared_error(y_r_test, mlp_pred)
    mlp_mse, mlp_r2 = mean_squared_error(y_r_test, mlp_pred), r2_score(y_r_test, mlp_pred)
    mlp_acc = (1 - mlp_mae / y_range) * 100

    print("\n" + "=" * 68)
    print("ALGORITHM EFFICIENCY TESTING & BENCHMARK SUMMARY (FULL 4-YEAR DATASET)")
    print(f"Total Valid Records: {len(X):,} 15-Minute Intervals (~2M+ raw rows)")
    print(f"Training Set (80%): {len(X_train):,} rows | Testing Set (20%): {len(X_test):,} rows")
    print("=" * 68)
    print(f"\n[TIER 1] Binary Peak Alert Detection (Logistic Regression | tau = {threshold_75:.4f} kW)")
    print(f"  Confusion Matrix ({tot:,} Test Records):")
    print(f"    True Negative (TN) : {tn:,} ({(tn/tot)*100:.2f}%)  [Correct Off-Peak]")
    print(f"    False Positive (FP): {fp:,} ({(fp/tot)*100:.2f}%)   [False Alarm]")
    print(f"    False Negative (FN): {fn:,} ({(fn/tot)*100:.2f}%)   [Missed Peak]")
    print(f"    True Positive (TP) : {tp:,} ({(tp/tot)*100:.2f}%)  [Correct Peak Alert]")
    print(f"  Scores: Acc = {lr_acc:.2f}% | Precision = {lr_prec:.2f}% | Recall = {lr_rec:.2f}%")
    print(f"          Specificity = {lr_spec:.2f}% | F1 = {lr_f1:.2f}% | ROC-AUC = {lr_auc:.2f}%")
    print(f"          MCC = +{lr_mcc:.4f} | Balanced Acc = {lr_bal:.2f}%")

    print(f"\n[TIER 2] Continuous Forecasting (Random Forest Regressor)")
    print(f"  Accuracy = {rf_acc:.2f}% | R² = {rf_r2:.4f} | MAE = {rf_mae:.4f} kW | RMSE = {rf_rmse:.4f} kW | MSE = {rf_mse:.4f} kW²")

    print(f"\n[TIER 3] Continuous Forecasting (ANN - MLPRegressor)")
    print(f"  Accuracy = {mlp_acc:.2f}% | R² = {mlp_r2:.4f} | MAE = {mlp_mae:.4f} kW | RMSE = {mlp_rmse:.4f} kW | MSE = {mlp_mse:.4f} kW²")
    print("=" * 68 + "\n")

    # 6. Generate Unified Visualization Canvas with Translucent Red Peak Alert Bars
    print("Generating Matplotlib reference chart with Logistic Regression peak alerts...")
    res_df = pd.DataFrame({"Actual": y_r_test, "RF_Pred": rf_pred, "ANN_Pred": mlp_pred, "LR_Pred": lr_pred}, index=X_test.index)
    daily_res = res_df.resample("D").mean()
    lr_threshold = daily_res["LR_Pred"].quantile(0.75)
    peak_days = daily_res[daily_res["LR_Pred"] >= lr_threshold]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.plot(daily_res.index, daily_res["Actual"], color="black", linewidth=1.8, label="Actual Household Power (Daily Mean kW)")
    ax.plot(daily_res.index, daily_res["RF_Pred"], color="#10b981", linestyle="--", linewidth=1.8, label="Random Forest Forecast (kW)")
    ax.plot(daily_res.index, daily_res["ANN_Pred"], color="#f59e0b", linestyle=":", linewidth=1.8, label="ANN (MLP) Forecast (kW)")

    delta = pd.Timedelta(days=1)
    for day in peak_days.index:
        ax.axvspan(day, day + delta, color="red", alpha=0.18, edgecolor=None)

    ax.axhline(baseline_mean, color="#3b82f6", linestyle=":", linewidth=1.5, label=f"Baseline Mean ({baseline_mean:.4f} kW)")
    ax.axhline(threshold_75, color="#ef4444", linestyle="-.", linewidth=1.5, label=f"75th Percentile Peak Cutoff ({threshold_75:.4f} kW)")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="red", alpha=0.18, label="Logistic Regression Peak Alert Zone"))
    labels.append("Logistic Regression Peak Alert Zone")
    ax.legend(handles=handles, labels=labels, loc="upper right", framealpha=0.95)

    ax.set_title("Multi-Model Household Power Forecast & Logistic Regression Peak Alert Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date & Time (UCI Individual Household Power Consumption Dataset)", fontsize=11)
    ax.set_ylabel("Global Active Power (kW)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_img = os.path.join(base_dir, "unified_dual_tier_forecast.png")
    plt.savefig(out_img)
    plt.close(fig)
    print(f"Unified visualization canvas saved to: {out_img}")

    # 7. Export Full Dataset to dataset.js
    print("Exporting full dataset to dataset.js...")
    df_export = df_15m.copy()
    df_export["Peak_Alert"] = (df_export[target_col] >= threshold_75).astype(int)

    records = [
        [str(r[0])[:16], round(float(r[1]), 4), round(float(r[2]), 2), round(float(r[3]), 2), round(float(r[4]), 2), round(float(r[5]), 2), int(r[6])]
        for r in df_export[["Global_active_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Peak_Alert"]].itertuples()
    ]
    stats = {
        "totalRecords": len(records), "startDate": records[0][0], "endDate": records[-1][0],
        "meanKW": round(baseline_mean, 4), "peakThreshold": round(threshold_75, 4),
        "minKW": round(float(df_15m[target_col].min()), 4), "maxKW": round(float(df_15m[target_col].max()), 4)
    }

    with open(os.path.join(base_dir, "dataset.js"), "w", encoding="utf-8") as f:
        f.write("window.DATASET_STATS = " + json.dumps(stats) + ";\n")
        f.write("window.DATASET_ROWS = " + json.dumps(records) + ";\n")

    # 8. Open Dashboard in Browser
    html_file = os.path.join(base_dir, "index.html")
    if os.path.exists(html_file):
        url = f"file:///{html_file.replace(os.sep, '/')}"
        print(f"Dashboard available at: {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return out_img

if __name__ == "__main__":
    run_pipeline()
