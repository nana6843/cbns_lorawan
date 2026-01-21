#!/usr/bin/env python3
"""
LoRaWAN ML-Based ADR - ENERGY-AWARE VERSION
Multi-Objective: PDR (Reliability) + Energy Efficiency
COMPARISON: Random Forest vs LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time  # ⭐ For inference time measurement
warnings.filterwarnings('ignore')

# ⭐ IMPORT LIGHTGBM
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
    LGBM_AVAILABLE = False

print("="*70)
print("LoRaWAN ML-Based ADR - Energy-Aware Configuration")
print("Multi-Objective: PDR + Energy Efficiency")
print("MODEL COMPARISON: Random Forest vs LightGBM")
print("="*70)

# ========== LOAD DATASET ==========

df = pd.read_csv('dataset_complete_with_rssi_snr.csv')
print(f"\n✓ Loaded {len(df)} records")
print(f"✓ Unique experiments: {df['experiment'].nunique()}")
print(f"✓ Unique nodes: {df['node_id'].nunique()}")

# ========== FEATURE ENGINEERING ==========
# [Keep all your existing feature engineering code - same as before]

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

print("✓ Created energy-aware features")

# ========== PATH LOSS MODEL ==========
# [Keep your path loss fitting code - same as before]

print("\n" + "="*70)
print("Fitting Path Loss Model from NS-3 Data...")
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

print(f"✓ Fitted Path Loss Model:")
print(f"  PL0: {PL0_fitted:.2f} dB")
print(f"  n:   {n_fitted:.2f}")

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

# ========== TRAIN RANDOM FOREST ==========

print("\n" + "="*70)
print("Training Random Forest Model...")
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

# Inference time (average over test set)
start_time = time.time()
_ = rf_model.predict(X_test)
rf_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

print(f"\n🌳 Random Forest Results:")
print(f"  Training time:   {rf_training_time:.2f}s")
print(f"  R² Score:        {r2_rf:.4f}")
print(f"  MAE:             {mae_rf:.2f}%")
print(f"  RMSE:            {rmse_rf:.2f}%")
print(f"  Inference time:  {rf_inference_time:.4f} ms/sample")

# Feature Importance
rf_importance = pd.DataFrame({
    'Feature': features_model,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# ========== TRAIN LIGHTGBM ==========

if LGBM_AVAILABLE:
    print("\n" + "="*70)
    print("Training LightGBM Model...")
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
        verbose=-1  # Suppress training output
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
    lgbm_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    print(f"\n⚡ LightGBM Results:")
    print(f"  Training time:   {lgbm_training_time:.2f}s")
    print(f"  R² Score:        {r2_lgbm:.4f}")
    print(f"  MAE:             {mae_lgbm:.2f}%")
    print(f"  RMSE:            {rmse_lgbm:.2f}%")
    print(f"  Inference time:  {lgbm_inference_time:.4f} ms/sample")
    
    # Feature Importance
    lgbm_importance = pd.DataFrame({
        'Feature': features_model,
        'Importance': lgbm_model.feature_importances_
    }).sort_values('Importance', ascending=False)

# ========== MODEL COMPARISON ==========

if LGBM_AVAILABLE:
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Metric': ['R² Score', 'MAE (%)', 'RMSE (%)', 
                   'Training Time (s)', 'Inference Time (ms)'],
        'Random Forest': [
            f"{r2_rf:.4f}",
            f"{mae_rf:.2f}",
            f"{rmse_rf:.2f}",
            f"{rf_training_time:.2f}",
            f"{rf_inference_time:.4f}"
        ],
        'LightGBM': [
            f"{r2_lgbm:.4f}",
            f"{mae_lgbm:.2f}",
            f"{rmse_lgbm:.2f}",
            f"{lgbm_training_time:.2f}",
            f"{lgbm_inference_time:.4f}"
        ],
        'Winner': [
            'RF' if r2_rf > r2_lgbm else 'LightGBM',
            'RF' if mae_rf < mae_lgbm else 'LightGBM',
            'RF' if rmse_rf < rmse_lgbm else 'LightGBM',
            'RF' if rf_training_time < lgbm_training_time else 'LightGBM',
            'RF' if rf_inference_time < lgbm_inference_time else 'LightGBM'
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Performance improvement
    r2_improvement = ((r2_lgbm - r2_rf) / r2_rf) * 100
    mae_improvement = ((mae_rf - mae_lgbm) / mae_rf) * 100
    speed_improvement = ((rf_inference_time - lgbm_inference_time) / rf_inference_time) * 100
    
    print(f"\n💡 Key Insights:")
    print(f"  R² improvement:       {r2_improvement:+.2f}%")
    print(f"  MAE improvement:      {mae_improvement:+.2f}%")
    print(f"  Inference speedup:    {speed_improvement:+.2f}%")
    
    # Recommendation
    print(f"\n🎯 Recommendation:")
    if abs(r2_improvement) < 1 and speed_improvement > 10:
        print(f"  → Use LightGBM: Similar accuracy, {speed_improvement:.1f}% faster")
    elif r2_improvement > 1:
        print(f"  → Use LightGBM: {r2_improvement:.1f}% better accuracy")
    else:
        print(f"  → Use Random Forest: Proven reliability, good performance")

# ========== VISUALIZATION ==========

if LGBM_AVAILABLE:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: RF Prediction
    axes[0, 0].scatter(y_test, y_pred_rf, alpha=0.6, s=50, edgecolors='black', c='blue')
    axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('True PDR (%)', fontsize=11)
    axes[0, 0].set_ylabel('Predicted PDR (%)', fontsize=11)
    axes[0, 0].set_title(f'Random Forest (R²={r2_rf:.4f})', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(5, 95, f'MAE: {mae_rf:.2f}%', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: LightGBM Prediction
    axes[0, 1].scatter(y_test, y_pred_lgbm, alpha=0.6, s=50, edgecolors='black', c='green')
    axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True PDR (%)', fontsize=11)
    axes[0, 1].set_ylabel('Predicted PDR (%)', fontsize=11)
    axes[0, 1].set_title(f'LightGBM (R²={r2_lgbm:.4f})', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(5, 95, f'MAE: {mae_lgbm:.2f}%', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Performance Comparison
    metrics = ['R²', 'MAE', 'RMSE']
    rf_metrics = [r2_rf, mae_rf, rmse_rf]
    lgbm_metrics = [r2_lgbm, mae_lgbm, rmse_lgbm]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for visualization (R² is 0-1, MAE/RMSE are %)
    rf_norm = [r2_rf * 100, mae_rf, rmse_rf]
    lgbm_norm = [r2_lgbm * 100, mae_lgbm, rmse_lgbm]
    
    axes[0, 2].bar(x - width/2, rf_norm, width, label='Random Forest', 
                   color='blue', edgecolor='black', alpha=0.7)
    axes[0, 2].bar(x + width/2, lgbm_norm, width, label='LightGBM', 
                   color='green', edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Metric', fontsize=11)
    axes[0, 2].set_ylabel('Value', fontsize=11)
    axes[0, 2].set_title('Performance Metrics Comparison', fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(['R² (×100)', 'MAE (%)', 'RMSE (%)'])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Feature Importance Comparison (Top 10)
    top_n = 10
    rf_top = rf_importance.head(top_n)
    lgbm_top = lgbm_importance.head(top_n)
    
    # Get union of top features
    top_features = list(set(rf_top['Feature'].tolist() + lgbm_top['Feature'].tolist()))[:top_n]
    
    rf_imp_dict = dict(zip(rf_importance['Feature'], rf_importance['Importance']))
    lgbm_imp_dict = dict(zip(lgbm_importance['Feature'], lgbm_importance['Importance']))
    
    rf_values = [rf_imp_dict.get(f, 0) for f in top_features]
    lgbm_values = [lgbm_imp_dict.get(f, 0) for f in top_features]
    
    y_pos = np.arange(len(top_features))
    axes[1, 0].barh(y_pos - 0.2, rf_values, 0.4, label='RF', 
                    color='blue', edgecolor='black', alpha=0.7)
    axes[1, 0].barh(y_pos + 0.2, lgbm_values, 0.4, label='LightGBM', 
                    color='green', edgecolor='black', alpha=0.7)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(top_features, fontsize=9)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Importance', fontsize=11)
    axes[1, 0].set_title('Feature Importance Comparison', fontweight='bold')
    axes[1, 0].legend()
    
    # Plot 5: Inference Time Comparison
    models = ['Random Forest', 'LightGBM']
    inference_times = [rf_inference_time, lgbm_inference_time]
    training_times = [rf_training_time, lgbm_training_time]
    
    x = np.arange(len(models))
    axes[1, 1].bar(x - 0.2, training_times, 0.4, label='Training (s)', 
                   color='orange', edgecolor='black', alpha=0.7)
    axes[1, 1].bar(x + 0.2, [t/1000 for t in inference_times], 0.4, 
                   label='Inference (s)', color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Model', fontsize=11)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=11)
    axes[1, 1].set_title('Training & Inference Time', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (train, infer) in enumerate(zip(training_times, inference_times)):
        axes[1, 1].text(i - 0.2, train, f'{train:.2f}s', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1, 1].text(i + 0.2, infer/1000, f'{infer:.3f}ms', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 6: Residual Comparison
    rf_residuals = y_test.values - y_pred_rf
    lgbm_residuals = y_test.values - y_pred_lgbm
    
    axes[1, 2].hist(rf_residuals, bins=50, alpha=0.6, label='RF', 
                    color='blue', edgecolor='black')
    axes[1, 2].hist(lgbm_residuals, bins=50, alpha=0.6, label='LightGBM', 
                    color='green', edgecolor='black')
    axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Prediction Error (%)', fontsize=11)
    axes[1, 2].set_ylabel('Frequency', fontsize=11)
    axes[1, 2].set_title('Residual Distribution', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ml_comparison_rf_vs_lightgbm.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison visualization: ml_comparison_rf_vs_lightgbm.png")

# ========== SAVE MODELS ==========

import joblib

# Save both models
joblib.dump(rf_model, 'lorawan_adr_model_rf.pkl')
print("\n✓ Saved: lorawan_adr_model_rf.pkl")

if LGBM_AVAILABLE:
    joblib.dump(lgbm_model, 'lorawan_adr_model_lgbm.pkl')
    print("✓ Saved: lorawan_adr_model_lgbm.pkl")

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

print("\n" + "="*70)
print("✅ Training Complete!")
print("="*70)

if LGBM_AVAILABLE:
    print(f"\n📊 Final Verdict:")
    if r2_lgbm > r2_rf:
        print(f"  🏆 LightGBM wins with {(r2_lgbm-r2_rf)*100:.2f}% better R²")
    else:
        print(f"  🏆 Random Forest wins with {(r2_rf-r2_lgbm)*100:.2f}% better R²")
    
    print(f"\n💰 Computational Efficiency:")
    print(f"  Training:  LightGBM is {rf_training_time/lgbm_training_time:.2f}x faster")
    print(f"  Inference: LightGBM is {rf_inference_time/lgbm_inference_time:.2f}x faster")

print("="*70)