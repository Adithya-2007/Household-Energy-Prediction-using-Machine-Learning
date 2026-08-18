# Household-Energy-Prediction-using-Machine-Learning

## 1. Project Overview & Documentation
This project uses Machine Learning (Logistic Regression) to forecast peak household energy demand based on historical power consumption patterns.

- **Course:** 23AID205 - AI / Machine Learning
- **Live Hosted Webpage:** https://YOUR_USERNAME.github.io/household-power-prediction/

---

## 2. Methodology & Implementation
1. **Data Preprocessing:** Resampled raw power records (~50,000 observations) to 15-minute mean intervals to smooth high-frequency noise.
2. **Thresholding:** Set a binary load classification target at the 75th percentile of power consumption.
3. **Feature Engineering:** Extracted cyclical sine/cosine time representations, autoregressive power lags (lag_1 through lag_96), rolling window statistics, and short-term rates of change.
4. **Model:** Trained a balanced `LogisticRegression` model using standard feature scaling (`StandardScaler`).

---

## 3. Test Cases & Validation Procedures
Validation is performed via an 80/20 chronological split. To verify model stability against stochastic variance, performance is tested across 6 random seeds (`10, 42, 100, 500, 1234, 2026`).

---

## 4. Demonstration Instructions (How to Run & Reproduce)

### Prerequisites
Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
