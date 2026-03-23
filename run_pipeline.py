"""
================================================================
Stock Market Analytics Engine — Master Pipeline Runner
================================================================
Run this single file to execute the complete pipeline:

    python run_pipeline.py

Steps:
    1  Data ingestion (yfinance API / simulated)
    2  Data cleaning & validation
    3  Technical indicator engineering (8 indicators)
    4  ML model training (XGBoost, TimeSeriesSplit)
    5  Combined signal generation (ML + Technical Rules)
    6  Portfolio backtesting ($100K per stock)
    7  Markowitz portfolio optimisation
    8  Chart generation (8 charts)
    9  Automated report output

Author  : Kartikesh Pal
Project : Stock Market Analytics & ML Prediction
Tools   : Python · yfinance · pandas · XGBoost · Plotly · Power BI
================================================================
"""

import os, sys, logging, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Create output directories
for d in ["data/raw","data/processed","models","output_charts","reports"]:
    os.makedirs(d, exist_ok=True)


def print_banner():
    print("\n" + "═"*62)
    print("  📈  Stock Market Analytics & ML Prediction Engine")
    print("  Python · XGBoost · yfinance · Power BI · SQL")
    print("═"*62 + "\n")


def main():
    print_banner()
    t0 = time.time()

    # ── Step 1: Data ──────────────────────────────────────────────
    log.info("STEP 1/8 — Data Pipeline")
    from data_pipeline import (generate_sample_data, fetch_stock_data,
                               clean_pipeline, validate_data, save_to_csv)
    try:
        raw = fetch_stock_data(period="2y")
        log.info("  ✓ Live data fetched from yfinance")
    except Exception as e:
        log.warning(f"  Live fetch failed ({e}). Using simulated data.")
        raw = generate_sample_data()

    clean  = clean_pipeline(raw)
    report = validate_data(clean)
    save_to_csv(clean)
    print(f"  ✓ {report['total_rows']:,} rows | {len(report['tickers'])} tickers | {report['date_range']}\n")

    # ── Step 2: Feature Engineering ───────────────────────────────
    log.info("STEP 2/8 — Feature Engineering (8 Indicators)")
    from feature_engineering import add_all_features, generate_rule_signals, feature_summary
    df = add_all_features(clean)
    df = generate_rule_signals(df)
    print(f"  ✓ {df.shape[1]} features computed\n")
    print("  Indicator Summary:")
    print(feature_summary(df).to_string())
    print()

    # ── Step 3: ML Model ──────────────────────────────────────────
    log.info("STEP 3/8 — ML Model Training (XGBoost + TimeSeriesSplit)")
    from ml_model import (build_target, train_and_evaluate, predict_signals,
                           save_model, feature_importance_report, portfolio_metrics)
    df_model = build_target(df)
    results  = train_and_evaluate(df_model)
    df       = predict_signals(results["model"], results["scaler"], df, results["feature_names"])
    save_model(results)
    print(f"\n  ✓ Model trained | Accuracy: {results['avg_accuracy']} "
          f"| AUC: {results['avg_auc']} | F1: {results['avg_f1']}\n")

    fi = feature_importance_report(results["model"], results["feature_names"])
    print("  Top 5 Features:")
    for _, row in fi.head(5).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<22} {bar} {row['importance']:.4f}")
    print()

    # ── Step 4: Combined Signals ──────────────────────────────────
    log.info("STEP 4/8 — Combined Signal Generation")
    from trading_signals import generate_combined_signals
    df = generate_combined_signals(df)
    sig_counts = df.groupby(["Ticker","Combined_Signal"]).size().unstack(fill_value=0)
    print("  Signal Distribution:")
    print(sig_counts.to_string())
    print()

    # ── Step 5: Backtesting ───────────────────────────────────────
    log.info("STEP 5/8 — Portfolio Backtesting ($100K per stock)")
    from trading_signals import backtest_portfolio, compute_correlation_matrix
    bt_summary = backtest_portfolio(df, capital_per_stock=100_000)
    print("\n  Backtest Summary:")
    print(bt_summary.to_string(index=False))
    print()

    # ── Step 6: Portfolio Metrics ─────────────────────────────────
    log.info("STEP 6/8 — Portfolio Risk Metrics")
    pm = portfolio_metrics(df)
    print("  Portfolio Metrics:")
    print(pm.to_string())
    print()

    # ── Step 7: Correlation ───────────────────────────────────────
    log.info("STEP 7/8 — Correlation Matrix")
    corr = compute_correlation_matrix(df)
    print("  Correlation Matrix:")
    print(corr.to_string())
    print()

    # ── Step 8: Charts ────────────────────────────────────────────
    log.info("STEP 8/8 — Generating Charts")
    from visualization import (plot_ytd_performance, plot_rsi, plot_macd,
                                plot_correlation_heatmap, plot_monthly_returns,
                                plot_risk_return, plot_candlestick, plot_feature_importance)
    plot_ytd_performance(df)
    plot_rsi(df, "AAPL")
    plot_macd(df, "AAPL")
    plot_correlation_heatmap(df)
    plot_monthly_returns(df, "AAPL")
    plot_risk_return(df)
    plot_candlestick(df, "AAPL")
    plot_feature_importance(fi)
    print("  ✓ 8 charts saved to output_charts/\n")

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("═"*62)
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print("═"*62)

    print("""
  📁 Output Files:
     data/processed/stock_data_clean.csv
     models/xgb_stock_model.pkl
     output_charts/ (8 PNG + HTML charts)

  📊 Power BI:
     Open powerbi/StockAnalytics.pbix
     Connect to: data/processed/stock_data_clean.csv

  📝 Next Steps:
     • Replace generate_sample_data() with fetch_stock_data()
       in production for live Yahoo Finance data
     • Set DATABASE_URL in .env for PostgreSQL storage
     • Schedule: python run_pipeline.py via cron / Task Scheduler
    """)


if __name__ == "__main__":
    main()
