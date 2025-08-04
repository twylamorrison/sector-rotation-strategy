# Sector Rotation Strategy (2015–2024)

This project implements a sector rotation strategy using 6-month momentum to allocate capital among U.S. sector ETFs.

## 📈 Features
- Momentum-based ranking of 11 SPDR sector ETFs
- Regime switching based on 200-day moving average of SPY
- Backtesting against SPY with transaction costs
- Performance attribution: returns by sector and average weights
- Rolling beta analysis and Sharpe ratio heatmap

## 🧪 Tech Stack
- Python
- Pandas, NumPy, yFinance, Matplotlib

## 🚀 How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
