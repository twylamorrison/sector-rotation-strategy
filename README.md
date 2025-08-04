# Sector Rotation Strategy (2015–2024)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/sector_rotation_strategy.ipynb)

This project implements a sector rotation strategy using 6-month momentum to allocate capital among U.S. sector ETFs.

---

## 📋 Features

- Momentum-based ranking of 11 SPDR sector ETFs  
- Regime switching based on 200-day moving average of SPY  
- Backtesting against SPY with transaction costs included  
- Performance attribution: sector-level returns and average portfolio weights  
- Rolling beta analysis and Sharpe ratio heatmap  

---

## 🧰 Tech Stack

- Python  
- Pandas, NumPy, yfinance, Matplotlib, tqdm  

---

## 🚀 How to Run

1. **Install requirements**:

   ```bash
   pip install -r requirements.txt
