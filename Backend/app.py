from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__, static_folder='static')
BASE = os.path.dirname(__file__)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
dt_risk  = joblib.load(f'{BASE}/model_decision_tree.pkl')
rf_risk  = joblib.load(f'{BASE}/model_rf_classifier.pkl')
knn_risk = joblib.load(f'{BASE}/model_knn.pkl')
rf_brent = joblib.load(f'{BASE}/model_rf_brent.pkl')
dir_lpg  = joblib.load(f'{BASE}/model_direction_lpg.pkl')
dir_pet  = joblib.load(f'{BASE}/model_direction_petrol.pkl')
dir_die  = joblib.load(f'{BASE}/model_direction_diesel.pkl')
km       = joblib.load(f'{BASE}/model_kmeans.pkl')
iso      = joblib.load(f'{BASE}/model_isolation_forest.pkl')

with open(f'{BASE}/model_stats.json') as f:
    stats = json.load(f)
with open(f'{BASE}/last_row.json') as f:
    last_row = json.load(f)

FEAT   = stats['feature_cols']
RS     = stats['raw_scale']
df_hist = pd.read_csv(f'{BASE}/history.csv')
df_full = pd.read_csv('ML_ready_data.csv')
df_full = df_full.drop(columns=[c for c in df_full.columns if 'Unnamed' in c])
df_full = df_full.sort_values('date').reset_index(drop=True)

RISK_NAMES  = ['Low', 'Medium', 'High', 'Critical']
DIR_NAMES   = ['Stable', 'Increase', 'Decrease']
print("All models loaded successfully.")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sc(val, key):
    s = RS[key]
    return float(np.clip((val - s['min']) / (s['max'] - s['min']), 0, 1))

def unsc(val, key):
    s = RS[key]
    return float(val * (s['max'] - s['min']) + s['min'])

def build_vector(brent, wti, inr, gpr, lpg, petrol, diesel):
    sb = sc(brent,  'brent_price')
    sw = sc(wti,    'wti_price')
    si = sc(inr,    'usd_rate')
    sg = sc(gpr,    'GPR')
    sl = sc(lpg,    'avg_lpg_price')
    sp = sc(petrol, 'avg_petrol')
    sd = sc(diesel, 'avg_diesel')
    lr = last_row

    row = {
        'brent_price': sb, 'wti_price': sw, 'usd_rate': si, 'GPR': sg,
        'brent_lag1':  lr['brent_lag1'], 'brent_lag3': lr['brent_lag3'], 'brent_lag6': lr['brent_lag6'],
        'wti_lag1':    lr['wti_lag1'],   'wti_lag3':   lr['wti_lag3'],
        'gpr_lag1':    lr['gpr_lag1'],   'gpr_lag3':   lr['gpr_lag3'],
        'usd_rate_lag1': lr['usd_rate_lag1'], 'usd_rate_lag3': lr['usd_rate_lag3'],
        'lpg_price_lag1': lr['lpg_price_lag1'], 'lpg_price_lag3': lr['lpg_price_lag3'],
        'petrol_price_lag1': lr['petrol_price_lag1'], 'petrol_price_lag3': lr['petrol_price_lag3'],
        'diesel_price_lag1': lr['diesel_price_lag1'], 'diesel_price_lag3': lr['diesel_price_lag3'],
        'brent_roll3': (sb + lr['brent_lag1'] + lr['brent_lag3']) / 3,
        'gpr_roll3':   (sg + lr['gpr_lag1']   + lr['gpr_lag3'])   / 3,
        'inr_roll3':   (si + lr['usd_rate_lag1'] + lr['usd_rate_lag3']) / 3,
        'brent_roll6': lr['brent_roll6'], 'gpr_roll6': lr['gpr_roll6'],
        'brent_pct_change': (sb - lr['brent_lag1']) / (lr['brent_lag1'] + 1e-9),
        'wti_pct_change':   (sw - lr['wti_lag1'])   / (lr['wti_lag1']   + 1e-9),
        'gpr_pct_change':   (sg - lr['gpr_lag1'])   / (lr['gpr_lag1']   + 1e-9),
        'inr_pct_change':   (si - lr['usd_rate_lag1']) / (lr['usd_rate_lag1'] + 1e-9),
        'lpg_pct_change':   (sl - lr['lpg_price_lag1']) / (lr['lpg_price_lag1'] + 1e-9),
        'petrol_pct_change': (sp - lr['petrol_price_lag1']) / (lr['petrol_price_lag1'] + 1e-9),
        'diesel_pct_change': (sd - lr['diesel_price_lag1']) / (lr['diesel_price_lag1'] + 1e-9),
        'crude_inr_impact': sb * si,
        'conflict_crude_index': sg * sb,
        'brent_wti_spread': sb - sw,
        'inr_gpr_stress': si * sg,
        'crude_inr_normalized': (sb * si) / (sl + 1e-9),
        'lpg_momentum': sl - lr['lpg_price_lag3'],
        'conflict_intensity': (
            (sg - df_full['GPR'].mean()) / df_full['GPR'].std() +
            (sb - df_full['brent_price'].mean()) / df_full['brent_price'].std()
        ),
    }
    return np.array([[row[f] for f in FEAT]])

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.json
    brent  = float(d['brent'])
    wti    = float(d['wti'])
    inr    = float(d['inr'])
    gpr    = float(d['gpr'])
    lpg    = float(d['lpg'])
    petrol = float(d['petrol'])
    diesel = float(d['diesel'])
    months = int(d.get('months', 3))
    fuel   = d.get('fuel', 'lpg')

    X = build_vector(brent, wti, inr, gpr, lpg, petrol, diesel)

    # ── 1. Risk Level (Decision Tree — 97.4% accurate) ────────────────────────
    risk_pred   = int(dt_risk.predict(X)[0])
    risk_proba  = dt_risk.predict_proba(X)[0].tolist()
    risk_rf     = int(rf_risk.predict(X)[0])
    risk_knn    = int(knn_risk.predict(X)[0])

    # ── 2. Price Direction (RF Classifier — 84-89% accurate) ─────────────────
    lpg_dir    = int(dir_lpg.predict(X)[0])
    petrol_dir = int(dir_pet.predict(X)[0])
    diesel_dir = int(dir_die.predict(X)[0])
    lpg_dir_p  = dir_lpg.predict_proba(X)[0].tolist()
    pet_dir_p  = dir_pet.predict_proba(X)[0].tolist()
    die_dir_p  = dir_die.predict_proba(X)[0].tolist()

    # ── 3. Brent Next Month (RF Regressor — R²=0.267) ─────────────────────────
    brent_next_scaled = float(rf_brent.predict(X)[0])
    brent_next_raw    = unsc(brent_next_scaled, 'brent_price')
    brent_pct_change  = (brent_next_raw - brent) / brent * 100

    # ── 4. KMeans Cluster ─────────────────────────────────────────────────────
    cluster = int(km.predict(X)[0])

    # ── 5. Anomaly Detection ──────────────────────────────────────────────────
    anomaly_score = float(iso.decision_function(X)[0])
    is_anomaly    = bool(iso.predict(X)[0] == -1)

    # ── 6. Fuel Price Estimate (data-driven, NOT fake formula) ────────────────
    # Based on direction + historical median change when direction is predicted
    fuel_col_map = {'lpg': 'avg_lpg_price', 'petrol': 'avg_petrol', 'diesel': 'avg_diesel'}
    fuel_dir_map = {'lpg': lpg_dir, 'petrol': petrol_dir, 'diesel': diesel_dir}
    cur_price    = {'lpg': lpg, 'petrol': petrol, 'diesel': diesel}[fuel]
    sel_dir      = fuel_dir_map[fuel]

    # Historical median % change per direction class from actual data
    fcol = fuel_col_map[fuel]
    df_full['pct'] = df_full[fcol].pct_change() * 100
    hist_increase = df_full[df_full['pct'] > 0.5]['pct'].median()
    hist_decrease = df_full[df_full['pct'] < -0.5]['pct'].median()
    hist_stable   = df_full[(df_full['pct'] >= -0.5) & (df_full['pct'] <= 0.5)]['pct'].median()

    # Forecast per month using predicted direction + historical median
    monthly_pct = {0: hist_stable, 1: hist_increase, 2: hist_decrease}
    month_pct   = monthly_pct.get(sel_dir, 0)

    # Risk multiplier — higher risk = amplify the change
    risk_mult = [0.5, 1.0, 1.5, 2.2][risk_pred]
    adj_pct   = month_pct * risk_mult

    forecast = []
    price = cur_price
    for m in range(1, months + 1):
        # Direction is for next month — subsequent months decay toward stable
        decay = 1.0 if m == 1 else max(0.3, 1.0 - (m-1)*0.2)
        change = price * (adj_pct / 100) * decay
        price  = round(price + change, 2)
        forecast.append({
            'month': m,
            'price': price,
            'change': round(change, 2),
            'pct': round((adj_pct * decay), 2)
        })

    # ── 7. KNN Similar Months ─────────────────────────────────────────────────
    sb = sc(brent, 'brent_price')
    sg = sc(gpr,   'GPR')
    si = sc(inr,   'usd_rate')
    sl = sc(lpg,   'avg_lpg_price')

    dists = []
    for _, row in df_full.iterrows():
        d_val = np.sqrt(
            (sg - row['GPR'])**2 * 2.0 +
            (sb - row['brent_price'])**2 * 1.5 +
            (si - row['usd_rate'])**2 * 1.0 +
            (sl - row['avg_lpg_price'])**2 * 0.5
        )
        dists.append(d_val)
    df_full['_dist'] = dists
    similar = df_full.nsmallest(6, '_dist').iloc[1:6]  # skip identical row

    similar_out = []
    for _, row in similar.iterrows():
        similar_out.append({
            'date':    row['date'],
            'gpr':     round(unsc(row['GPR'], 'GPR'), 1),
            'brent':   round(unsc(row['brent_price'], 'brent_price'), 1),
            'lpg':     round(unsc(row['avg_lpg_price'], 'avg_lpg_price'), 0),
            'petrol':  round(unsc(row['avg_petrol'], 'avg_petrol'), 1),
            'diesel':  round(unsc(row['avg_diesel'], 'avg_diesel'), 1),
            'risk':    int(row['risk_level']),
            'is_conflict': int(row.get('conflict_event', 0)),
        })

    # ── 8. Historical Chart Data ───────────────────────────────────────────────
    hist_out = []
    for _, row in df_full.tail(24).iterrows():
        hist_out.append({
            'date':   row['date'][:7],
            'gpr':    round(unsc(row['GPR'], 'GPR'), 1),
            'brent':  round(unsc(row['brent_price'], 'brent_price'), 1),
            'lpg':    round(unsc(row['avg_lpg_price'], 'avg_lpg_price'), 0),
            'petrol': round(unsc(row['avg_petrol'], 'avg_petrol'), 1),
            'diesel': round(unsc(row['avg_diesel'], 'avg_diesel'), 1),
            'risk':   int(row['risk_level']),
        })

    return jsonify({
        'risk': {
            'level': risk_pred,
            'name':  RISK_NAMES[risk_pred],
            'proba': [round(p*100,1) for p in risk_proba],
            'rf':    RISK_NAMES[risk_rf],
            'knn':   RISK_NAMES[risk_knn],
        },
        'direction': {
            'lpg':    {'label': DIR_NAMES[lpg_dir],    'proba': [round(p*100,1) for p in lpg_dir_p]},
            'petrol': {'label': DIR_NAMES[petrol_dir], 'proba': [round(p*100,1) for p in pet_dir_p]},
            'diesel': {'label': DIR_NAMES[diesel_dir], 'proba': [round(p*100,1) for p in die_dir_p]},
        },
        'brent_next': {
            'price': round(brent_next_raw, 2),
            'pct':   round(brent_pct_change, 2),
            'r2':    0.267,
        },
        'cluster': cluster,
        'anomaly': {
            'score':      round(anomaly_score, 4),
            'is_anomaly': is_anomaly,
        },
        'forecast': forecast,
        'similar':  similar_out,
        'history':  hist_out,
        'inputs': {
            'brent': brent, 'wti': wti, 'inr': inr, 'gpr': gpr,
            'lpg': lpg, 'petrol': petrol, 'diesel': diesel,
            'fuel': fuel, 'months': months,
        },
        'model_note': {
            'risk_accuracy':      '97.4% (Decision Tree)',
            'direction_accuracy': '84-89% (Random Forest)',
            'brent_r2':           '0.267 (RF Regressor)',
        }
    })

@app.route('/api/history')
def history():
    out = []
    for _, row in df_full.iterrows():
        out.append({
            'date':   row['date'][:7],
            'gpr':    round(unsc(row['GPR'], 'GPR'), 1),
            'brent':  round(unsc(row['brent_price'], 'brent_price'), 1),
            'lpg':    round(unsc(row['avg_lpg_price'], 'avg_lpg_price'), 0),
            'petrol': round(unsc(row['avg_petrol'], 'avg_petrol'), 1),
            'diesel': round(unsc(row['avg_diesel'], 'avg_diesel'), 1),
            'risk':   int(row['risk_level']),
            'conflict': int(row.get('conflict_event', 0)),
        })
    return jsonify(out)

if __name__ == '__main__':
    app.run(debug=False, port=5000)