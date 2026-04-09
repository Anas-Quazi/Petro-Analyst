# PetroAnalyst-India 🛢️📊
### Geopolitical Risk-Based Oil & LPG Price Prediction System for India

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

**PetroAnalyst-India** is a data science project that predicts and detects future hikes in crude oil and LPG prices in India caused by escalating geopolitical conflicts in the Middle East — particularly US-Iran tensions. Using historical market data, macroeconomic indicators, and geopolitical risk indices, the system acts as an **early warning mechanism** for policymakers and the Government of India's Ministry of Petroleum & Natural Gas.

---

## 🎯 Problem Statement

India imports approximately 85% of its crude oil. When geopolitical conflicts escalate in the Middle East — especially involving Iran, Iraq, or the Strait of Hormuz — global oil supply is disrupted, Brent crude prices spike, and India faces a direct economic impact through rising LPG and fuel prices.

The goal of this project is to build a **data-driven early warning system** that:
- Detects signals of an upcoming oil/LPG price hike **before** it hits the market
- Predicts price movements for the **next 30, 60, and 90 days**
- Classifies the current geopolitical situation into actionable **risk levels**
- Assists government policymakers in making **proactive fuel subsidy and import decisions**

---

## 🗂️ Project Pipeline

```
Problem Domain
      ↓
Dataset Collection
      ↓
Data Preprocessing        ← Clean each dataset individually
      ↓
Data Integration          ← Merge into one master dataset
      ↓
     EDA                  ← Explore + visualize patterns
      ↓
Data Analysis             ← Statistical & correlation analysis
      ↓
ML Algorithms             ← Regression, Classification, Clustering, KNN, Decision Tree
      ↓
Advanced ML               ← Neural Networks, Ensemble Methods
      ↓
Anomaly Detection         ← Spike & supply disruption alerts
      ↓
Data Visualization        ← Final results & insights
      ↓
Outcomes & Key Takeaways
```

---

## 📦 Datasets

| # | Dataset | Source |
|---|---------|--------|
| 1 | Brent & WTI Crude Oil Historical Prices | [Kaggle](https://www.kaggle.com/datasets/nikitamanaenkov/historical-crude-oil-futures-prices-wti-and-brent) |
| 2 | LPG Cylinder Price History — India | [Kaggle](https://www.kaggle.com/datasets/imumerfarooq/lpg-cylinder-price-for-last-ten-years) |
| 3 | Geopolitical Risk Index (GPR) | [matteoiacoviello.com](https://www.matteoiacoviello.com/gpr.htm) |
| 4 | INR/USD Exchange Rate History | [Kaggle](https://www.kaggle.com/datasets/imbikramsaha/usd-inr-data) |
| 5 | India Crude Oil Import Prices | [PPAC — Govt. of India](https://ppac.gov.in/prices/international-prices-of-crude-oil) |

---

## 🧠 Data Mining Techniques Applied

| Technique | Purpose |
|-----------|---------|
| **Regression Analysis** | Predict future LPG & crude oil prices |
| **Classification** | Classify geopolitical situation into Low / Medium / High / Critical risk |
| **Clustering (K-Means)** | Group countries by geopolitical risk & India's import dependency |
| **KNN** | Match current crisis to similar historical events |
| **Decision Tree** | Trigger policy alerts based on price thresholds |
| **Correlation Analysis** | Find relationships between conflict events and price movements |
| **Anomaly Detection** | Detect abnormal price spikes in real time |
| **Neural Networks** | Analyze complex multi-variable patterns for long-term forecasting |
| **Association Rule Mining** | Discover hidden event-price relationships |
| **Sentiment Analysis (NLP)** | Mine news data for early conflict escalation signals |

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Dataset | Kaggle / PPAC / GPR Official |
| Data Handling | NumPy, Pandas |
| ML Algorithms | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Code Editor | VS Code + Jupyter Notebook |
| Version Control | Git & GitHub |
| AI Assistant | Claude (Anthropic) |

---

## 📁 Repository Structure

```
PetroAnalyst-India/
│
├── datasets/                  # Raw downloaded datasets
│   ├── brent_wti_crude.csv
│   ├── lpg_price_india.csv
│   ├── gpr_index.xlsx
│   ├── inr_usd_rate.csv
│   └── india_oil_imports.csv
│
├── notebooks/                 # Jupyter notebooks (one per phase)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_integration.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_data_analysis.ipynb
│   ├── 05_ml_algorithms.ipynb
│   ├── 06_advanced_ml.ipynb
│   ├── 07_anomaly_detection.ipynb
│   └── 08_visualization_outcomes.ipynb
│
├── outputs/                   # Saved plots, results, predictions
│
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/PetroAnalyst-India.git
cd PetroAnalyst-India
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
Place all downloaded datasets inside the `datasets/` folder as shown in the structure above.

### 4. Run notebooks
Open Jupyter Notebook in VS Code and run notebooks in order from `01` to `08`.

---

## 📊 Expected Outcomes

- Early detection of oil & LPG price hikes before they hit Indian markets
- 30 / 60 / 90 day price forecasting using historical and geopolitical data
- Geopolitical risk classification (Low / Medium / High / Critical)
- Anomaly alerts for sudden price spikes or supply disruptions
- Data-driven insights to support government fuel subsidy decisions
- Strategic oil reserve management recommendations

---

## 👤 Author

**Anas Quazi**
- GitHub: [@Anas-Quazi](https://github.com/Anas-Quazi)

---

## 📄 License

This project is licensed under the MIT License.

---

> *"Data is the new oil — and this project uses data to predict the oil."*
