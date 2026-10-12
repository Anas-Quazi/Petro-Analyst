"""
Run this ONCE to generate all required files:
  - model_stats.json
  - last_row.json  
  - history.csv
  - all .pkl model files

Usage: python setup.py
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               IsolationForest, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
import joblib, json, os

# ── CONFIG ────────────────────────────────────────────────────────────────────
# PUT YOUR ML_ready_data.csv PATH HERE
DATA_PATH = 'ML_ready_data.csv'
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

# ── BUILD TARGETS ─────────────────────────────────────────────────────────────
df['next_brent'] = df['brent_price'].shift(-1)
for col, name in [('avg_lpg_price','lpg'),('avg_petrol','petrol'),('avg_diesel','diesel')]:
    nxt = df[col].shift(-1)
    pct = (nxt - df[col]) / (df[col] + 1e-9) * 100
    df[f'next_{name}_dir'] = np.where(pct > 0.5, 1, np.where(pct < -0.5, 2, 0))
df = df.dropna(subset=['next_brent']).reset_index(drop=True)

# ── FEATURE COLUMNS ───────────────────────────────────────────────────────────
exclude = (
    ['date','risk_level','conflict_event',
     'avg_lpg_price','avg_petrol','avg_diesel',
     'next_brent','next_lpg_dir','next_petrol_dir','next_diesel_dir'] +
    [c for c in df.columns if 'outlier' in c]
)
FEAT = [c for c in df.columns if c not in exclude]
print(f"  Features: {len(FEAT)}")

X      = df[FEAT]
y_risk = df['risk_level']
split  = int(len(df) * 0.80)
X_tr, X_te         = X.iloc[:split],  X.iloc[split:]
y_r_tr, y_r_te     = y_risk.iloc[:split], y_risk.iloc[split:]
y_b_tr, y_b_te     = df['next_brent'].iloc[:split], df['next_brent'].iloc[split:]

# ── TRAIN MODELS ──────────────────────────────────────────────────────────────
print("\nTraining models...")

# Decision Tree — Risk (97.4%)
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3,
                             class_weight='balanced', random_state=42)
dt.fit(X_tr, y_r_tr)
acc = accuracy_score(y_r_te, dt.predict(X_te))
print(f"  Decision Tree (risk)   : {acc*100:.1f}% accuracy")
joblib.dump(dt, f'{OUT_DIR}/dt_risk.pkl')

# Random Forest Classifier — Risk
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight='balanced', random_state=42)
rf_clf.fit(X_tr, y_r_tr)
acc2 = accuracy_score(y_r_te, rf_clf.predict(X_te))
print(f"  RF Classifier (risk)   : {acc2*100:.1f}% accuracy")
joblib.dump(rf_clf, f'{OUT_DIR}/rf_risk.pkl')

# KNN — Risk
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_tr, y_r_tr)
acc3 = accuracy_score(y_r_te, knn.predict(X_te))
print(f"  KNN (risk)             : {acc3*100:.1f}% accuracy")
joblib.dump(knn, f'{OUT_DIR}/knn_risk.pkl')

# RF Regressor — Brent
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_reg.fit(X_tr, y_b_tr)
r2 = r2_score(y_b_te, rf_reg.predict(X_te))
print(f"  RF Regressor (brent)   : R²={r2:.3f}")
joblib.dump(rf_reg, f'{OUT_DIR}/rf_brent.pkl')

# Direction classifiers — LPG, Petrol, Diesel
for name in ['lpg','petrol','diesel']:
    col = f'next_{name}_dir'
    y_tr2 = df[col].iloc[:split]
    y_te2 = df[col].iloc[split:]
    m = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    m.fit(X_tr, y_tr2)
    acc4 = accuracy_score(y_te2, m.predict(X_te))
    print(f"  Direction ({name:<7})    : {acc4*100:.1f}% accuracy")
    joblib.dump(m, f'{OUT_DIR}/dir_{name}.pkl')

# KMeans Clustering
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X)
print(f"  KMeans clusters        : 4 groups")
joblib.dump(km, f'{OUT_DIR}/kmeans.pkl')

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
iso.fit(X)
n_anom = (iso.predict(X) == -1).sum()
print(f"  Isolation Forest       : {n_anom} anomalies detected")
joblib.dump(iso, f'{OUT_DIR}/isolation.pkl')

# ── SAVE model_stats.json ─────────────────────────────────────────────────────
# These are the SCALED min/max values from your ML_ready_data.csv
# Used to convert raw user input → scaled values for model prediction
raw_scale = {
    'brent_price'   : {'min': 27.0,   'max': 133.18},
    'wti_price'     : {'min': 16.0,   'max': 130.50},
    'usd_rate'      : {'min': 44.0,   'max': 87.0  },
    'GPR'           : {'min': 47.0,   'max': 324.0 },
    'avg_lpg_price' : {'min': 345.0,  'max': 1053.0},
    'avg_petrol'    : {'min': 47.0,   'max': 110.0 },
    'avg_diesel'    : {'min': 35.0,   'max': 98.0  },
}
stats = {
    'feature_cols': FEAT,
    'raw_scale':    raw_scale,
    'n_features':   len(FEAT),
    'train_rows':   split,
    'test_rows':    len(df) - split,
    'date_range':   f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}"
}
with open(f'{OUT_DIR}/model_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\n  model_stats.json saved")

# ── SAVE last_row.json ────────────────────────────────────────────────────────
# Last row of training data — used as lag reference when user gives current inputs
last_row = df.iloc[-1][FEAT].to_dict()
# Convert numpy types to Python native for JSON serialization
last_row = {k: float(v) for k, v in last_row.items()}
with open(f'{OUT_DIR}/last_row.json', 'w') as f:
    json.dump(last_row, f, indent=2)
print(f"  last_row.json saved")

# ── SAVE history.csv ─────────────────────────────────────────────────────────
# Historical data for charts and KNN similarity search
hist_cols = ['date','GPR','brent_price','wti_price','usd_rate',
             'avg_lpg_price','avg_petrol','avg_diesel',
             'risk_level','conflict_event']
hist_cols = [c for c in hist_cols if c in df.columns]
df[hist_cols].to_csv(f'{OUT_DIR}/history.csv', index=False)
print(f"  history.csv saved ({len(df)} rows)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  ALL FILES GENERATED SUCCESSFULLY")
print("="*50)
print("\nFiles created in:", OUT_DIR)
files = ['dt_risk.pkl','rf_risk.pkl','knn_risk.pkl','rf_brent.pkl',
         'dir_lpg.pkl','dir_petrol.pkl','dir_diesel.pkl',
         'kmeans.pkl','isolation.pkl',
         'model_stats.json','last_row.json','history.csv']
for f in files:
    path = f'{OUT_DIR}/{f}'
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"  {'✅' if size > 0 else '❌'} {f:<30} {size/1024:.1f} KB")

print("\nNext step: python app.py")
