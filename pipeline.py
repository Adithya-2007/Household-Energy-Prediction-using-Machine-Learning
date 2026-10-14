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

def eval_reg(name, y_true, y_pred, y_range):
    mae, rmse = mean_absolute_error(y_true, y_pred), root_mean_squared_error(y_true, y_pred)
    mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
    acc = (1 - mae / y_range) * 100
    print(f"\n[{name}] Continuous Forecasting")
    print(f"  Accuracy = {acc:.2f}% | R² = {r2:.4f} | MAE = {mae:.4f} kW | RMSE = {rmse:.4f} kW | MSE = {mse:.4f} kW²")
    return mae, rmse, mse, r2, acc

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "full_household_power_15m.csv")
    raw_txt_path = os.path.join(base_dir, "..", "household_power_consumption.txt")

    # Ingest 15-minute or raw UCI dataset
    if os.path.exists(csv_path):
        print("Loading full dataset from full_household_power_15m.csv...")
        df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    elif os.path.exists(raw_txt_path):
        print("Parsing raw UCI household_power_consumption.txt (2M+ records)...")
        raw = pd.read_csv(raw_txt_path, sep=";", na_values=["?"], low_memory=False)
        raw["Datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        raw = raw.dropna(subset=["Datetime", "Global_active_power"]).sort_values("Datetime").set_index("Datetime")
        cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        df = raw[cols].resample("15min").mean().dropna()
        df.to_csv(csv_path)
    else:
        raise FileNotFoundError("Could not locate full dataset file.")

    target = "Global_active_power"
    threshold_75 = float(df[target].quantile(0.75))
    baseline_mean = float(df[target].mean())
    y_reg_raw = df[target]
    y_class_raw = (df[target] >= threshold_75).astype(int)

    # Feature engineering (temporal, momentum, novelty lags)
    print("Engineering temporal, momentum, and sub-metering features...")
    X = pd.DataFrame(index=df.index)
    X["sin_hour"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    X["cos_hour"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    X["dayofweek"] = df.index.dayofweek
    X["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    for lag in [1, 2, 3, 4, 8, 96]:
        X[f"lag_{lag}"] = df[target].shift(lag)
    X["diff_1"] = df[target].shift(1) - df[target].shift(2)

    for w in [4, 96]:
        X[f"rolling_mean_{w}"] = df[target].shift(1).rolling(w).mean()
        X[f"rolling_std_{w}"] = df[target].shift(1).rolling(w).std()

    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Voltage", "Global_reactive_power"]:
        if col in df.columns:
            for lag in [1, 96]:
                X[f"{col}_lag_{lag}"] = df[col].shift(lag)

    valid_idx = X.dropna().index
    X, y_reg, y_class = X.loc[valid_idx], y_reg_raw.loc[valid_idx], y_class_raw.loc[valid_idx]

    # Chronological 80/20 train/test split & scaling
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_r_train, y_r_test = y_reg.iloc[:train_size], y_reg.iloc[train_size:]
    y_c_train, y_c_test = y_class.iloc[:train_size], y_class.iloc[train_size:]
    print(f"Dataset split: Train = {len(X_train):,} intervals (80%) | Test = {len(X_test):,} intervals (20%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training: Tier 1 (LR), Tier 2 (RF), Tier 3 (ANN)
    print("Training Tier 1: Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_c_train)
    lr_pred, lr_prob = lr.predict(X_test_scaled), lr.predict_proba(X_test_scaled)[:, 1]

    print("Training Tier 2: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(X_train, y_r_train)
    rf_pred = rf.predict(X_test)

    print("Training Tier 3: ANN (MLPRegressor)...")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True, max_iter=300, random_state=42).fit(X_train_scaled, y_r_train)
    mlp_pred = mlp.predict(X_test_scaled)

    # Benchmarking evaluation
    cm = confusion_matrix(y_c_test, lr_pred)
    tn, fp, fn, tp = cm.ravel()
    tot, y_range = len(y_c_test), y_r_test.max() - y_r_test.min()

    lr_acc = accuracy_score(y_c_test, lr_pred) * 100
    lr_auc = roc_auc_score(y_c_test, lr_prob) * 100
    lr_prec = precision_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_rec = recall_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_spec = (tn / (tn + fp)) * 100
    lr_f1 = f1_score(y_c_test, lr_pred, zero_division=0) * 100
    lr_mcc = matthews_corrcoef(y_c_test, lr_pred)
    lr_bal = balanced_accuracy_score(y_c_test, lr_pred) * 100

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

    eval_reg("TIER 2: Random Forest", y_r_test, rf_pred, y_range)
    eval_reg("TIER 3: ANN (MLPRegressor)", y_r_test, mlp_pred, y_range)
    print("=" * 68 + "\n")

    # Generate unified forecast canvas with translucent red peak alert bars
    print("Generating Matplotlib reference chart with Logistic Regression peak alerts...")
    res_df = pd.DataFrame({"Actual": y_r_test, "RF_Pred": rf_pred, "ANN_Pred": mlp_pred, "LR_Pred": lr_pred}, index=X_test.index)
    daily_res = res_df.resample("D").mean()
    peak_days = daily_res[daily_res["LR_Pred"] >= daily_res["LR_Pred"].quantile(0.75)]

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

    # Export full dataset to dataset.js
    print("Exporting full dataset to dataset.js...")
    df_export = df.copy()
    df_export["Peak_Alert"] = (df_export[target] >= threshold_75).astype(int)
    cols_export = ["Global_active_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Peak_Alert"]

    records = [
        [str(r[0])[:16], round(float(r[1]), 4), round(float(r[2]), 2), round(float(r[3]), 2), round(float(r[4]), 2), round(float(r[5]), 2), int(r[6])]
        for r in df_export[cols_export].itertuples()
    ]
    stats = {
        "totalRecords": len(records), "startDate": records[0][0], "endDate": records[-1][0],
        "meanKW": round(baseline_mean, 4), "peakThreshold": round(threshold_75, 4),
        "minKW": round(float(df[target].min()), 4), "maxKW": round(float(df[target].max()), 4)
    }

    with open(os.path.join(base_dir, "dataset.js"), "w", encoding="utf-8") as f:
        f.write("window.DATASET_STATS = " + json.dumps(stats) + ";\n")
        f.write("window.DATASET_ROWS = " + json.dumps(records) + ";\n")

    # Launch dashboard in browser
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
