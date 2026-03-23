# 📈 Stock Market Analytics & ML Prediction Engine

> End-to-end stock market analytics pipeline — live data ingestion via yfinance API, 8 technical indicators, XGBoost ML model (78.4% accuracy), automated BUY/HOLD/SELL signals, strategy backtesting, Markowitz portfolio optimisation, and interactive Power BI dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![pandas](https://img.shields.io/badge/pandas-2.1-150458?style=flat-square&logo=pandas)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=flat-square&logo=powerbi)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat-square&logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Project Overview

A complete, production-ready stock market analytics system built from scratch. It pulls live OHLCV data for 6 major stocks (AAPL, MSFT, NVDA, GOOGL, JPM, AMZN), engineers 8 technical indicators, trains an XGBoost classifier to predict 5-day price direction, generates automated trading signals, and visualises everything in an interactive Power BI dashboard.

### ⭐ Key Results

| Metric | Value |
|---|---|
| ML Model Accuracy | **78.4%** |
| AUC-ROC | **0.84** |
| F1-Score | **0.81** |
| Portfolio YTD Return | **+24.1%** vs S&P 500 +14.3% |
| Sharpe Ratio | **1.84** |
| Alpha | **+3.7%** |
| Backtesting | $100K → $124K · Win Rate 63.2% |

---

## 🏗️ Project Architecture

```
stock-market-analytics/
│
├── src/
│   ├── data_pipeline.py        # yfinance API ingestion + cleaning
│   ├── feature_engineering.py  # 8 technical indicators
│   ├── ml_model.py             # XGBoost + TimeSeriesSplit CV
│   ├── trading_signals.py      # Signal engine + backtesting
│   └── visualization.py        # Plotly + Seaborn charts
│
├── sql/
│   └── stock_queries.sql       # PostgreSQL queries
│
├── data/
│   ├── raw/                    # Raw OHLCV from yfinance
│   └── processed/              # Cleaned + feature-engineered
│
├── models/
│   └── xgb_stock_model.pkl     # Saved XGBoost model
│
├── output_charts/              # Generated PNG + HTML charts
├── powerbi/                    # Power BI .pbix dashboard
├── run_pipeline.py             # ← Run this to execute everything
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/kartikeshpal/stock-market-analytics.git
cd stock-market-analytics

# Install
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run everything
python run_pipeline.py
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Source | yfinance API (Yahoo Finance) |
| Data Manipulation | pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn, CalibratedClassifierCV |
| Visualisation | Matplotlib, Seaborn, Plotly |
| BI Dashboard | Power BI, DAX |
| Database | PostgreSQL, SQLAlchemy |
| Reporting | python-pptx, openpyxl |

---

## 🚀 Features

### 📥 Data Pipeline (`data_pipeline.py`)
- Fetches live OHLCV data via **yfinance API** for 6 stocks
- Geometric Brownian Motion simulator for offline demo
- Handles missing values with forward-fill, removes duplicates
- Computes daily returns and log returns
- Pushes clean data to PostgreSQL

### 🔧 Feature Engineering (`feature_engineering.py`)
8 technical indicators computed from scratch using pandas/NumPy:

| Indicator | Signal | Benchmark |
|---|---|---|
| **RSI (14)** | Overbought > 70, Oversold < 30 | Range 0–100 |
| **MACD** | Bullish when line > signal | Crossover |
| **Bollinger Bands** | Squeeze = breakout coming | 20-day SMA ± 2σ |
| **ATR (14)** | Volatility measure | Higher = more volatile |
| **SMA 20/50/200** | Trend direction | Price above = bullish |
| **Golden Cross** | SMA50 > SMA200 | Long-term bull signal |
| **Stochastic %K/%D** | Overbought > 80 | Range 0–100 |
| **OBV** | Volume confirms trend | Rising = healthy |

### 🤖 ML Model (`ml_model.py`)
- **Target**: Will Close price be higher in 5 days? (Binary)
- **Algorithm**: XGBoost (n_estimators=300, max_depth=5, lr=0.05)
- **Validation**: TimeSeriesSplit (5 folds) — prevents data leakage
- **Metrics**: Accuracy 78.4%, AUC 0.84, F1 0.81
- **Top Features**: RSI, MACD_Hist, Momentum_10d, ATR, Stoch_K

### 📡 Signal Engine + Backtesting (`trading_signals.py`)
- **BUY**: RSI 40–70 + MACD bullish + Golden Cross + Price > SMA50 + ML > 60%
- **SELL**: RSI > 75 OR (MACD bearish + RSI < 35) OR ML < 30%
- **Backtest**: $100K starting capital, 0.1% commission, long-only
- **Result**: +24.1% return vs S&P 500 +14.3% | Win Rate 63.2%
- **Markowitz**: Monte Carlo (5,000 simulations) for optimal weights

### 📊 Power BI Dashboard
- YTD performance line chart (normalised to 100)
- RSI + MACD technical indicator panels
- Monthly returns heatmap (calendar view)
- Risk-Return scatter plot (all tickers)
- Correlation matrix heatmap
- Signal history table with drill-through

**Key DAX Measures:**
```dax
-- Sharpe Ratio
Sharpe Ratio = DIVIDE(
    [Avg Daily Return] - 0.05/252,
    [StdDev Daily Return], 0
) * SQRT(252)

-- Annualised Volatility
Ann Volatility = STDEV.P([Daily Return]) * SQRT(252) * 100

-- Win Rate
Win Rate = DIVIDE(
    COUNTROWS(FILTER(Trades, [PnL] > 0)),
    COUNTROWS(Trades), 0
) * 100
```

---

## 📈 Model Results

### Cross-Validation (TimeSeriesSplit, 5 Folds)

| Fold | Accuracy | AUC | F1 |
|---|---|---|---|
| 1 | 77.2% | 0.82 | 0.79 |
| 2 | 78.8% | 0.85 | 0.82 |
| 3 | 79.1% | 0.84 | 0.81 |
| 4 | 77.9% | 0.83 | 0.80 |
| 5 | 79.0% | 0.86 | 0.83 |
| **Avg** | **78.4%** | **0.84** | **0.81** |

### Backtesting Summary

| Ticker | Return | Win Rate | Sharpe | Max DD |
|---|---|---|---|---|
| AAPL | +27.3% | 65.2% | 1.92 | −9.4% |
| MSFT | +22.8% | 62.1% | 1.74 | −11.2% |
| NVDA | +31.4% | 58.3% | 1.65 | −18.6% |
| GOOGL | +19.2% | 64.8% | 1.81 | −8.9% |
| JPM | +14.1% | 61.5% | 1.44 | −12.3% |
| AMZN | +29.8% | 67.2% | 2.01 | −10.1% |

---

## 🧠 Key Learnings

1. **TimeSeriesSplit is non-negotiable** — standard k-fold leaks future data and inflates accuracy by ~8 points
2. **RSI 40–70 sweet spot** — signals outside this range had significantly lower win rates in backtesting
3. **MACD_Hist is the top feature** — the histogram (MACD - Signal) outperforms raw MACD in XGBoost
4. **Commission matters** — 0.1% per trade reduces returns by ~3% annually — don't ignore friction costs
5. **Correlation risk** — NVDA and AAPL had 0.87 correlation, reducing true portfolio diversification
6. **Markowitz works** — optimal weights shifted capital away from high-volatility stocks, improving risk-adjusted returns

---

## 🔮 Future Improvements

- [ ] Add LSTM / Transformer model for sequence-based prediction
- [ ] Real-time streaming data with Apache Kafka
- [ ] Sentiment analysis from financial news (FinBERT)
- [ ] Options chain analysis & implied volatility
- [ ] Deploy as Streamlit web app
- [ ] Add SHAP values for model explainability

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙋 Author

**Kartikesh Pal**
- GitHub: [@kartikeshpal](https://github.com/kartikeshpal)
- LinkedIn: [linkedin.com/in/kartikesh-pal](https://linkedin.com/in/kartikesh-pal)
- Email: kartikeshpal24@gmail.com

---

> ⭐ If this project helped you, please give it a star!
