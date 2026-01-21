#!/usr/bin/env python3
"""
LoRaWAN ML-Based ADR - COMPLETE MODEL COMPARISON
Train & Compare: Random Forest, LightGBM, XGBoost
Multi-Objective: PDR (Reliability) + Energy Efficiency
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import joblib
warnings.filterwarnings('ignore')

# ⭐ IMPORT GRADIENT BOOSTING LIBRARIES
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

print("="*70)
print("LoRaWAN ML-Based ADR - COMPLETE MODEL COMPARISON")
print("Models: Random Forest | LightGBM | XGBoost")
print("Multi-Objective: PDR + Energy Efficiency")
print("="*70)

# ========== LOAD DATASET ==========

df = pd.read_csv('dataset_complete_with_rssi_snr.csv')
print(f"\n✓ Loaded {len(df)} records")
print(f"✓ Unique experiments: {df['experiment'].nunique()}")
print(f"✓ Unique nodes: {df['node_id'].nunique()}")

# ========== FEATURE ENGINEERING ==========

print("\n" + "="*70)
print("Feature Engineering...")
print("="*70)

# 1. Link Quality Features
df['link_margin_dB'] = df['RSSI'] - df['Sensitivity']
df['snr_margin_dB'] = df['SNR'] - (-7.5)
df['is_above_sensitivity'] = df['Above_Sensitivity'].astype(int)

# 2. Efficiency Metrics
df['throughput_ratio'] = df['RxPackets'] / df['TxPackets']
df['packet_loss_rate'] = df['PacketsLost'] / df['TxPackets']

# 3. Spatial Features
df['distance_normalized'] = df['distance'] / df['distance'].max()
df['radial_distance'] = np.sqrt(df['x']**2 + df['y']**2)

# 4. Power Efficiency
df['power_distance_ratio'] = df['txPower'] / (df['distance'] + 1)

# 5. Energy Features
df['EnergyPerPacket_mJ'] = df['EnergyPerPacket_J'] * 1000

def calculate_energy_from_params(txPower_dBm, toa_ms):
    txPower_mW = 10 ** (txPower_dBm / 10.0)
    toa_s = toa_ms / 1000.0
    return txPower_mW * toa_s

df['Calculated_Energy_mJ'] = df.apply(
    lambda row: calculate_energy_from_params(row['txPower'], row['AvgToA_ms']), 
    axis=1
)

# 6. Multi-objective score
df['pdr_normalized'] = df['PDR'] / 100.0
max_energy = df['EnergyPerPacket_mJ'].max()
df['energy_efficiency'] = 1 - (df['EnergyPerPacket_mJ'] / max_energy)
df['performance_score'] = 0.6 * df['pdr_normalized'] + 0.4 * df['energy_efficiency']

print("✓ Created 14 features for energy-aware prediction")

# ========== PATH LOSS MODEL ==========

print("\n" + "="*70)
print("Fitting Path Loss Model...")
print("="*70)

df['path_loss_dB'] = df['txPower'] - df['RSSI']

from scipy.optimize import curve_fit

def path_loss_model(distance, PL0, n):
    d0 = 1.0
    return PL0 + 10 * n * np.log10(distance / d0)

distances = df['distance'].values
path_losses = df['path_loss_dB'].values

params, covariance = curve_fit(path_loss_model, distances, path_losses, p0=[7.7, 3.76])
PL0_fitted, n_fitted = params

print(f"✓ Path Loss Model: PL0={PL0_fitted:.2f} dB, n={n_fitted:.2f}")

# ========== NS-3 PARAMETERS ==========

NOISE_FLOOR_DBM = -117.03

SENSITIVITY_MAP = {
    0: -142.5, 1: -140.0, 2: -137.5, 
    3: -135.0, 4: -132.5, 5: -130.0
}

TOA_LOOKUP = {
    0: 991.232, 1: 495.616, 2: 247.808, 
    3: 128.0, 4: 66.048, 5: 34.048
}

def build_energy_lookup():
    energy_lookup = {}
    for dr in [0, 1, 2, 3, 4, 5]:
        energy_lookup[dr] = {}
        for power in [6, 8, 10, 12, 14]:
            toa_ms = TOA_LOOKUP[dr]
            energy_mJ = calculate_energy_from_params(power, toa_ms)
            energy_lookup[dr][power] = energy_mJ
    return energy_lookup

ENERGY_LOOKUP = build_energy_lookup()

# ========== PREPARE TRAINING DATA ==========

features_model = [
    'distance',
    'distance_normalized',
    'radial_distance',
    'dr',
    'freq',
    'txPower',
    'RSSI',
    'SNR',
    'Sensitivity',
    'link_margin_dB',
    'snr_margin_dB',
    'is_above_sensitivity',
    'power_distance_ratio',
    'EnergyPerPacket_mJ'
]

X = df[features_model]
y_pdr = df['PDR']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pdr, test_size=0.2, random_state=42
)

print(f"\n✓ Training set: {len(X_train)} samples")
print(f"✓ Test set:     {len(X_test)} samples")

# ========== STORAGE FOR RESULTS ==========

results = {}

# ========== TRAIN RANDOM FOREST ==========

print("\n" + "="*70)
print("1️⃣  Training Random Forest...")
print("="*70)

start_time = time.time()

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_training_time = time.time() - start_time

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Metrics
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Inference time
start_time = time.time()
_ = rf_model.predict(X_test)
rf_inference_time = (time.time() - start_time) / len(X_test) * 1000

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': features_model,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🌳 Random Forest Results:")
print(f"  Training time:   {rf_training_time:.2f}s")
print(f"  R² Score:        {r2_rf:.4f}")
print(f"  MAE:             {mae_rf:.2f}%")
print(f"  RMSE:            {rmse_rf:.2f}%")
print(f"  Inference time:  {rf_inference_time:.4f} ms/sample")

results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'r2': r2_rf,
    'mae': mae_rf,
    'rmse': rmse_rf,
    'train_time': rf_training_time,
    'inference_time': rf_inference_time,
    'importance': rf_importance
}

# ========== TRAIN LIGHTGBM ==========

if LGBM_AVAILABLE:
    print("\n" + "="*70)
    print("2️⃣  Training LightGBM...")
    print("="*70)
    
    start_time = time.time()
    
    lgbm_model = LGBMRegressor(
        n_estimators=200,
        max_depth=25,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgbm_model.fit(X_train, y_train)
    lgbm_training_time = time.time() - start_time
    
    # Predictions
    y_pred_lgbm = lgbm_model.predict(X_test)
    
    # Metrics
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    
    # Inference time
    start_time = time.time()
    _ = lgbm_model.predict(X_test)
    lgbm_inference_time = (time.time() - start_time) / len(X_test) * 1000
    
    # Feature importance
    lgbm_importance = pd.DataFrame({
        'Feature': features_model,
        'Importance': lgbm_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n⚡ LightGBM Results:")
    print(f"  Training time:   {lgbm_training_time:.2f}s")
    print(f"  R² Score:        {r2_lgbm:.4f}")
    print(f"  MAE:             {mae_lgbm:.2f}%")
    print(f"  RMSE:            {rmse_lgbm:.2f}%")
    print(f"  Inference time:  {lgbm_inference_time:.4f} ms/sample")
    
    results['LightGBM'] = {
        'model': lgbm_model,
        'predictions': y_pred_lgbm,
        'r2': r2_lgbm,
        'mae': mae_lgbm,
        'rmse': rmse_lgbm,
        'train_time': lgbm_training_time,
        'inference_time': lgbm_inference_time,
        'importance': lgbm_importance
    }

# ========== TRAIN XGBOOST ==========

if XGBOOST_AVAILABLE:
    print("\n" + "="*70)
    print("3️⃣  Training XGBoost...")
    print("="*70)
    
    start_time = time.time()
    
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=25,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_training_time = time.time() - start_time
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Metrics
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    
    # Inference time
    start_time = time.time()
    _ = xgb_model.predict(X_test)
    xgb_inference_time = (time.time() - start_time) / len(X_test) * 1000
    
    # Feature importance
    xgb_importance = pd.DataFrame({
        'Feature': features_model,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🚀 XGBoost Results:")
    print(f"  Training time:   {xgb_training_time:.2f}s")
    print(f"  R² Score:        {r2_xgb:.4f}")
    print(f"  MAE:             {mae_xgb:.2f}%")
    print(f"  RMSE:            {rmse_xgb:.2f}%")
    print(f"  Inference time:  {xgb_inference_time:.4f} ms/sample")
    
    results['XGBoost'] = {
        'model': xgb_model,
        'predictions': y_pred_xgb,
        'r2': r2_xgb,
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'train_time': xgb_training_time,
        'inference_time': xgb_inference_time,
        'importance': xgb_importance
    }

# ========== COMPREHENSIVE COMPARISON ==========

print("\n" + "="*70)
print("📊 COMPREHENSIVE MODEL COMPARISON")
print("="*70)

# Build comparison table
comparison_data = []
for model_name, model_results in results.items():
    comparison_data.append({
        'Model': model_name,
        'R²': f"{model_results['r2']:.4f}",
        'MAE (%)': f"{model_results['mae']:.2f}",
        'RMSE (%)': f"{model_results['rmse']:.2f}",
        'Train Time (s)': f"{model_results['train_time']:.2f}",
        'Inference (ms)': f"{model_results['inference_time']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Find best model for each metric
best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
best_mae = min(results.items(), key=lambda x: x[1]['mae'])
best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])
best_speed = min(results.items(), key=lambda x: x[1]['inference_time'])

print("\n" + "="*70)
print("🏆 WINNERS BY METRIC")
print("="*70)
print(f"  Best R² Score:       {best_r2[0]} ({best_r2[1]['r2']:.4f})")
print(f"  Best MAE:            {best_mae[0]} ({best_mae[1]['mae']:.2f}%)")
print(f"  Best RMSE:           {best_rmse[0]} ({best_rmse[1]['rmse']:.2f}%)")
print(f"  Fastest Inference:   {best_speed[0]} ({best_speed[1]['inference_time']:.4f} ms)")

# Overall recommendation
if len(results) > 1:
    # Calculate composite score (R² weight: 60%, Speed weight: 40%)
    composite_scores = {}
    max_r2 = max([r['r2'] for r in results.values()])
    min_inference = min([r['inference_time'] for r in results.values()])
    
    for model_name, model_results in results.items():
        r2_score_norm = model_results['r2'] / max_r2
        speed_score = min_inference / model_results['inference_time']
        composite = 0.6 * r2_score_norm + 0.4 * speed_score
        composite_scores[model_name] = composite
    
    best_overall = max(composite_scores.items(), key=lambda x: x[1])
    
    print(f"\n🎯 RECOMMENDED MODEL: {best_overall[0]}")
    print(f"   Composite Score: {best_overall[1]:.4f}")
    print(f"   (60% accuracy + 40% speed)")

# ========== VISUALIZATION ==========

num_models = len(results)
fig_rows = 2
fig_cols = max(3, num_models)
fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(6*fig_cols, 10))

if num_models == 1:
    axes = np.array([axes])

# Row 1: Prediction scatter plots for each model
colors = ['blue', 'green', 'red']
for idx, (model_name, model_results) in enumerate(results.items()):
    col = idx
    ax = axes[0, col] if fig_rows > 1 else axes[col]
    
    ax.scatter(y_test, model_results['predictions'], 
               alpha=0.6, s=50, edgecolors='black', c=colors[idx])
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2)
    ax.set_xlabel('True PDR (%)', fontsize=11)
    ax.set_ylabel('Predicted PDR (%)', fontsize=11)
    ax.set_title(f'{model_name}\n(R²={model_results["r2"]:.4f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(5, 95, f'MAE: {model_results["mae"]:.2f}%', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide unused subplots in row 1
for idx in range(num_models, fig_cols):
    axes[0, idx].axis('off')

# Row 2: Comparison charts
if len(results) > 1:
    # Plot 1: R² Comparison
    ax1 = axes[1, 0]
    model_names = list(results.keys())
    r2_values = [results[m]['r2'] for m in model_names]
    bars = ax1.bar(model_names, r2_values, color=colors[:len(model_names)], 
                   edgecolor='black', alpha=0.7)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² Score Comparison', fontweight='bold')
    ax1.set_ylim([min(r2_values)*0.95, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MAE & RMSE Comparison
    ax2 = axes[1, 1]
    x = np.arange(len(model_names))
    width = 0.35
    mae_values = [results[m]['mae'] for m in model_names]
    rmse_values = [results[m]['rmse'] for m in model_names]
    
    ax2.bar(x - width/2, mae_values, width, label='MAE', 
            color='orange', edgecolor='black', alpha=0.7)
    ax2.bar(x + width/2, rmse_values, width, label='RMSE', 
            color='purple', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Error (%)', fontsize=11)
    ax2.set_title('Error Metrics Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Inference Time Comparison
    ax3 = axes[1, 2]
    train_times = [results[m]['train_time'] for m in model_names]
    inference_times = [results[m]['inference_time'] for m in model_names]
    
    x = np.arange(len(model_names))
    ax3.bar(x - width/2, train_times, width, label='Training (s)', 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax3_right = ax3.twinx()
    ax3_right.bar(x + width/2, inference_times, width, label='Inference (ms)', 
                  color='crimson', edgecolor='black', alpha=0.7)
    
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('Training Time (s)', fontsize=11, color='steelblue')
    ax3_right.set_ylabel('Inference Time (ms)', fontsize=11, color='crimson')
    ax3.set_title('Training & Inference Time', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3_right.tick_params(axis='y', labelcolor='crimson')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Hide remaining subplots in row 2
    for idx in range(3, fig_cols):
        axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('ml_comparison_all_models.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: ml_comparison_all_models.png")

# ========== FEATURE IMPORTANCE COMPARISON ==========

if len(results) > 1:
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Get top 10 features from best model
    top_features = list(results[best_overall[0]]['importance'].head(10)['Feature'])
    
    x = np.arange(len(top_features))
    width = 0.8 / len(results)
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        importance_dict = dict(zip(model_results['importance']['Feature'], 
                                  model_results['importance']['Importance']))
        values = [importance_dict.get(f, 0) for f in top_features]
        
        offset = (idx - len(results)/2 + 0.5) * width
        ax.barh(x + offset, values, width, label=model_name, 
                color=colors[idx], edgecolor='black', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance Comparison (Top 10)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved feature importance: feature_importance_comparison.png")

# ========== SAVE ALL MODELS ==========

print("\n" + "="*70)
print("Saving Models...")
print("="*70)

for model_name, model_results in results.items():
    filename = f'lorawan_adr_model_{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model_results['model'], filename)
    print(f"✓ Saved: {filename}")

# Save parameters
path_loss_params = {
    'PL0': PL0_fitted,
    'n': n_fitted,
    'noise_floor_dBm': NOISE_FLOOR_DBM,
    'sensitivity_map': SENSITIVITY_MAP,
    'toa_lookup': TOA_LOOKUP,
    'energy_lookup': ENERGY_LOOKUP
}
joblib.dump(path_loss_params, 'path_loss_params.pkl')
print("✓ Saved: path_loss_params.pkl")

# ========== PERFORMANCE SUMMARY TABLE (FOR PAPER) ==========

print("\n" + "="*70)
print("📄 PERFORMANCE SUMMARY TABLE (for paper)")
print("="*70)

summary_table = pd.DataFrame({
    'Model': list(results.keys()),
    'R²': [f"{results[m]['r2']:.4f}" for m in results.keys()],
    'MAE (%)': [f"{results[m]['mae']:.2f}" for m in results.keys()],
    'RMSE (%)': [f"{results[m]['rmse']:.2f}" for m in results.keys()],
    'Training Time (s)': [f"{results[m]['train_time']:.2f}" for m in results.keys()],
    'Inference Time (ms)': [f"{results[m]['inference_time']:.4f}" for m in results.keys()]
})

print("\n" + summary_table.to_string(index=False))

# Export to CSV for easy copy-paste to paper
summary_table.to_csv('model_comparison_summary.csv', index=False)
print("\n✓ Exported to: model_comparison_summary.csv")

# ========== FINAL SUMMARY ==========

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

if len(results) > 1:
    print(f"\n🏆 Best Overall Model: {best_overall[0]}")
    print(f"   R²: {results[best_overall[0]]['r2']:.4f}")
    print(f"   Inference: {results[best_overall[0]]['inference_time']:.4f} ms")
    
    # Performance gains
    rf_result = results['Random Forest']
    best_result = results[best_overall[0]]
    
    if best_overall[0] != 'Random Forest':
        r2_gain = ((best_result['r2'] - rf_result['r2']) / rf_result['r2']) * 100
        speed_gain = ((rf_result['inference_time'] - best_result['inference_time']) / 
                     rf_result['inference_time']) * 100
        
        print(f"\n💡 Gains over Random Forest:")
        print(f"   R² improvement: {r2_gain:+.2f}%")
        print(f"   Speed improvement: {speed_gain:+.2f}%")

print("\n📁 Generated Files:")
print("   - ml_comparison_all_models.png")
print("   - feature_importance_comparison.png")
print("   - model_comparison_summary.csv")
for m in results.keys():
    print(f"   - lorawan_adr_model_{m.lower().replace(' ', '_')}.pkl")
print("   - path_loss_params.pkl")

print("\n" + "="*70)