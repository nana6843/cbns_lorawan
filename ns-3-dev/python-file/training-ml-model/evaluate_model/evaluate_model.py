#!/usr/bin/env python3
"""
LoRaWAN Model Performance Evaluation - SEPARATE IMAGES
Comprehensive evaluation of Random Forest vs LightGBM models
Generates individual image files for each plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error, explained_variance_score)
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("LoRaWAN Model Performance Evaluation - SEPARATE IMAGES")
print("Random Forest vs LightGBM Comparison")
print("="*70)

# ========== LOAD MODELS ==========

print("\n📦 Loading trained models...")

try:
    rf_model = joblib.load('lorawan_adr_model_random_forest.pkl')
    print("✓ Loaded Random Forest model")
except FileNotFoundError:
    print("❌ Random Forest model not found!")
    exit(1)

try:
    lgbm_model = joblib.load('lorawan_adr_model_lightgbm.pkl')
    print("✓ Loaded LightGBM model")
    lgbm_available = True
except FileNotFoundError:
    print("⚠️  LightGBM model not found - will evaluate RF only")
    lgbm_available = False

try:
    params = joblib.load('path_loss_params.pkl')
    print("✓ Loaded parameters")
except FileNotFoundError:
    print("⚠️  Parameters file not found")

# ========== LOAD & PREPARE DATA ==========

print("\n📊 Loading dataset...")

df = pd.read_csv('dataset_complete_with_rssi_snr.csv')
print(f"✓ Loaded {len(df)} records")

# Feature Engineering
df['link_margin_dB'] = df['RSSI'] - df['Sensitivity']
df['snr_margin_dB'] = df['SNR'] - (-7.5)
df['is_above_sensitivity'] = df['Above_Sensitivity'].astype(int)
df['throughput_ratio'] = df['RxPackets'] / df['TxPackets']
df['packet_loss_rate'] = df['PacketsLost'] / df['TxPackets']
df['distance_normalized'] = df['distance'] / df['distance'].max()
df['radial_distance'] = np.sqrt(df['x']**2 + df['y']**2)
df['power_distance_ratio'] = df['txPower'] / (df['distance'] + 1)
df['EnergyPerPacket_mJ'] = df['EnergyPerPacket_J'] * 1000

features = [
    'distance', 'distance_normalized', 'radial_distance', 'dr', 'freq',
    'txPower', 'RSSI', 'SNR', 'Sensitivity', 'link_margin_dB',
    'snr_margin_dB', 'is_above_sensitivity', 'power_distance_ratio',
    'EnergyPerPacket_mJ'
]

X = df[features]
y = df['PDR']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Test set: {len(X_test)} samples")

# ========== EVALUATION FUNCTIONS ==========

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive performance metrics"""
    metrics = {
        'Model': model_name,
        'R²': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'Max Error': np.max(np.abs(y_true - y_pred)),
        'Mean Error': np.mean(y_true - y_pred),
        'Std Error': np.std(y_true - y_pred)
    }
    return metrics

def evaluate_by_sf(y_true, y_pred, sf_values, model_name):
    """Evaluate performance by Spreading Factor"""
    sf_results = []
    for sf in sorted(sf_values.unique()):
        mask = sf_values == sf
        if mask.sum() > 0:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            count = mask.sum()
            sf_results.append({
                'Model': model_name,
                'SF': sf,
                'R²': r2,
                'MAE': mae,
                'Count': count
            })
    return pd.DataFrame(sf_results)

def evaluate_by_distance(y_true, y_pred, distances, model_name):
    """Evaluate performance by distance ranges"""
    bins = [0, 1000, 2000, 3000, 4000, 5000, 10000]
    labels = ['0-1km', '1-2km', '2-3km', '3-4km', '4-5km', '5km+']
    distance_ranges = pd.cut(distances, bins=bins, labels=labels)
    
    dist_results = []
    for dist_range in labels:
        mask = distance_ranges == dist_range
        if mask.sum() > 0:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            count = mask.sum()
            dist_results.append({
                'Model': model_name,
                'Distance Range': dist_range,
                'R²': r2,
                'MAE': mae,
                'Count': count
            })
    return pd.DataFrame(dist_results)

# ========== PREDICTIONS ==========

print("\n🔮 Generating predictions...")

# Random Forest
start = time.time()
y_pred_rf = rf_model.predict(X_test)
rf_inference_time = (time.time() - start) / len(X_test) * 1000
print(f"✓ Random Forest predictions complete ({rf_inference_time:.4f} ms/sample)")

# LightGBM
if lgbm_available:
    start = time.time()
    y_pred_lgbm = lgbm_model.predict(X_test)
    lgbm_inference_time = (time.time() - start) / len(X_test) * 1000
    print(f"✓ LightGBM predictions complete ({lgbm_inference_time:.4f} ms/sample)")

# ========== CALCULATE METRICS ==========

print("\n📈 Calculating performance metrics...")

rf_metrics = calculate_metrics(y_test, y_pred_rf, 'Random Forest')
print("\n🌳 Random Forest Performance:")
for key, value in rf_metrics.items():
    if key != 'Model':
        if key in ['R²', 'Explained Variance']:
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value:.2f}%")

if lgbm_available:
    lgbm_metrics = calculate_metrics(y_test, y_pred_lgbm, 'LightGBM')
    print("\n⚡ LightGBM Performance:")
    for key, value in lgbm_metrics.items():
        if key != 'Model':
            if key in ['R²', 'Explained Variance']:
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value:.2f}%")

# Prepare test dataframe
test_df = X_test.copy()
test_df['True_PDR'] = y_test.values
test_df['SF'] = test_df['dr'].map({0:12, 1:11, 2:10, 3:9, 4:8, 5:7})

# Evaluate by SF
rf_sf_perf = evaluate_by_sf(y_test.values, y_pred_rf, test_df['SF'], 'Random Forest')
if lgbm_available:
    lgbm_sf_perf = evaluate_by_sf(y_test.values, y_pred_lgbm, test_df['SF'], 'LightGBM')

# Evaluate by distance
rf_dist_perf = evaluate_by_distance(y_test.values, y_pred_rf, test_df['distance'], 'Random Forest')
if lgbm_available:
    lgbm_dist_perf = evaluate_by_distance(y_test.values, y_pred_lgbm, test_df['distance'], 'LightGBM')

# Calculate residuals
residuals_rf = y_test.values - y_pred_rf
if lgbm_available:
    residuals_lgbm = y_test.values - y_pred_lgbm

# ========== RANDOM FOREST VISUALIZATIONS ==========

print("\n🎨 Creating Random Forest visualizations...")

# 1. RF Prediction Scatter
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred_rf, alpha=0.5, s=50, edgecolors='black', linewidths=0.5, color='steelblue')
ax.plot([0, 100], [0, 100], 'r--', linewidth=3, label='Perfect Prediction')
ax.set_xlabel('True PDR (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted PDR (%)', fontsize=14, fontweight='bold')
ax.set_title(f'Random Forest: Prediction Accuracy\nR² = {rf_metrics["R²"]:.4f}', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.text(5, 95, f'MAE: {rf_metrics["MAE"]:.2f}%\nRMSE: {rf_metrics["RMSE"]:.2f}%',
        fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig('rf_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_prediction_scatter.png")
plt.close()

# 2. RF Residual Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_pred_rf, residuals_rf, alpha=0.5, s=50, edgecolors='black', linewidths=0.5, color='coral')
ax.axhline(y=0, color='red', linestyle='--', linewidth=3)
ax.set_xlabel('Predicted PDR (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Residuals (%)', fontsize=14, fontweight='bold')
ax.set_title('Random Forest: Residual Analysis', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.text(5, max(residuals_rf)*0.9, 
        f'Mean: {np.mean(residuals_rf):.2f}%\nStd: {np.std(residuals_rf):.2f}%',
        fontsize=13, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
plt.tight_layout()
plt.savefig('rf_residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_residual_plot.png")
plt.close()

# 3. RF Error Distribution
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(residuals_rf, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=3, label='Zero Error')
ax.set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Random Forest: Error Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('rf_error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_error_distribution.png")
plt.close()

# 4. RF Performance by SF
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.bar(rf_sf_perf['SF'], rf_sf_perf['R²'], color='steelblue', 
              edgecolor='black', alpha=0.8, width=0.6)
ax.set_xlabel('Spreading Factor', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('Random Forest: R² by Spreading Factor', fontsize=16, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')
for i, (sf, r2) in enumerate(zip(rf_sf_perf['SF'], rf_sf_perf['R²'])):
    ax.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('rf_performance_by_sf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_performance_by_sf.png")
plt.close()

# 5. RF MAE by Distance
fig, ax = plt.subplots(figsize=(10, 8))
x_pos = np.arange(len(rf_dist_perf))
bars = ax.bar(x_pos, rf_dist_perf['MAE'], color='coral', edgecolor='black', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(rf_dist_perf['Distance Range'], rotation=45, ha='right')
ax.set_xlabel('Distance Range', fontsize=14, fontweight='bold')
ax.set_ylabel('MAE (%)', fontsize=14, fontweight='bold')
ax.set_title('Random Forest: MAE by Distance', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, mae in enumerate(rf_dist_perf['MAE']):
    ax.text(i, mae + 0.2, f'{mae:.2f}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('rf_mae_by_distance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_mae_by_distance.png")
plt.close()

# 6. RF Feature Importance
fig, ax = plt.subplots(figsize=(10, 10))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

bars = ax.barh(range(len(feature_importance)), feature_importance['Importance'], 
               color='green', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['Feature'], fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
ax.set_title('Random Forest: Top 10 Features', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, imp in enumerate(feature_importance['Importance']):
    ax.text(imp, i, f' {imp:.4f}', va='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_feature_importance.png")
plt.close()

# ========== LIGHTGBM VISUALIZATIONS ==========

if lgbm_available:
    print("\n🎨 Creating LightGBM visualizations...")
    
    # 1. LightGBM Prediction Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_pred_lgbm, alpha=0.5, s=50, edgecolors='black', 
               linewidths=0.5, color='green')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=3, label='Perfect Prediction')
    ax.set_xlabel('True PDR (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted PDR (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'LightGBM: Prediction Accuracy\nR² = {lgbm_metrics["R²"]:.4f}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.text(5, 95, f'MAE: {lgbm_metrics["MAE"]:.2f}%\nRMSE: {lgbm_metrics["RMSE"]:.2f}%',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('lgbm_prediction_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_prediction_scatter.png")
    plt.close()
    
    # 2. LightGBM Residual Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_pred_lgbm, residuals_lgbm, alpha=0.5, s=50, edgecolors='black', 
               linewidths=0.5, color='lightgreen')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=3)
    ax.set_xlabel('Predicted PDR (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Residuals (%)', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM: Residual Analysis', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(5, max(residuals_lgbm)*0.9, 
            f'Mean: {np.mean(residuals_lgbm):.2f}%\nStd: {np.std(residuals_lgbm):.2f}%',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    plt.savefig('lgbm_residual_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_residual_plot.png")
    plt.close()
    
    # 3. LightGBM Error Distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(residuals_lgbm, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=3, label='Zero Error')
    ax.set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM: Error Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lgbm_error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_error_distribution.png")
    plt.close()
    
    # 4. LightGBM Performance by SF
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(lgbm_sf_perf['SF'], lgbm_sf_perf['R²'], color='green', 
                  edgecolor='black', alpha=0.8, width=0.6)
    ax.set_xlabel('Spreading Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM: R² by Spreading Factor', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (sf, r2) in enumerate(zip(lgbm_sf_perf['SF'], lgbm_sf_perf['R²'])):
        ax.text(i, r2 + 0.02, f'{r2:.3f}', ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('lgbm_performance_by_sf.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_performance_by_sf.png")
    plt.close()
    
    # 5. LightGBM MAE by Distance
    fig, ax = plt.subplots(figsize=(10, 8))
    x_pos = np.arange(len(lgbm_dist_perf))
    bars = ax.bar(x_pos, lgbm_dist_perf['MAE'], color='lightgreen', edgecolor='black', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lgbm_dist_perf['Distance Range'], rotation=45, ha='right')
    ax.set_xlabel('Distance Range', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (%)', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM: MAE by Distance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, mae in enumerate(lgbm_dist_perf['MAE']):
        ax.text(i, mae + 0.2, f'{mae:.2f}', ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig('lgbm_mae_by_distance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_mae_by_distance.png")
    plt.close()
    
    # 6. LightGBM Feature Importance
    fig, ax = plt.subplots(figsize=(10, 10))
    feature_importance_lgbm = pd.DataFrame({
        'Feature': features,
        'Importance': lgbm_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    bars = ax.barh(range(len(feature_importance_lgbm)), feature_importance_lgbm['Importance'], 
                   color='darkgreen', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(feature_importance_lgbm)))
    ax.set_yticklabels(feature_importance_lgbm['Feature'], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM: Top 10 Features', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, imp in enumerate(feature_importance_lgbm['Importance']):
        ax.text(imp, i, f' {imp:.4f}', va='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig('lgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lgbm_feature_importance.png")
    plt.close()

# ========== COMPARISON VISUALIZATIONS ==========

if lgbm_available:
    print("\n🎨 Creating comparison visualizations...")
    
    # Calculate improvements
    improvements = {
        'R² Improvement': ((lgbm_metrics['R²'] - rf_metrics['R²']) / rf_metrics['R²']) * 100,
        'MAE Improvement': ((rf_metrics['MAE'] - lgbm_metrics['MAE']) / rf_metrics['MAE']) * 100,
        'RMSE Improvement': ((rf_metrics['RMSE'] - lgbm_metrics['RMSE']) / rf_metrics['RMSE']) * 100,
        'Speed Improvement': ((rf_inference_time - lgbm_inference_time) / rf_inference_time) * 100
    }
    
    # 1. Comparison - Predictions
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_pred_rf, alpha=0.4, s=50, label='Random Forest', color='blue')
    ax.scatter(y_test, y_pred_lgbm, alpha=0.4, s=50, label='LightGBM', color='green')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=3, label='Perfect Prediction')
    ax.set_xlabel('True PDR (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted PDR (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Prediction Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_predictions.png")
    plt.close()
    
    # 2. Comparison - Metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_compare = ['R²', 'MAE', 'RMSE']
    rf_vals = [rf_metrics['R²']*100, rf_metrics['MAE'], rf_metrics['RMSE']]
    lgbm_vals = [lgbm_metrics['R²']*100, lgbm_metrics['MAE'], lgbm_metrics['RMSE']]
    
    x = np.arange(len(metrics_compare))
    width = 0.35
    
    ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='blue', 
           edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, lgbm_vals, width, label='LightGBM', color='green', 
           edgecolor='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['R² (×100)', 'MAE (%)', 'RMSE (%)'], fontsize=13)
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Performance Metrics', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (rf_val, lgbm_val) in enumerate(zip(rf_vals, lgbm_vals)):
        ax.text(i - width/2, rf_val, f'{rf_val:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
        ax.text(i + width/2, lgbm_val, f'{lgbm_val:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_metrics.png")
    plt.close()
    
    # 3. Comparison - Inference Speed
    fig, ax = plt.subplots(figsize=(10, 8))
    models = ['Random Forest', 'LightGBM']
    speeds = [rf_inference_time, lgbm_inference_time]
    colors_speed = ['blue', 'green']
    
    bars = ax.bar(models, speeds, color=colors_speed, edgecolor='black', alpha=0.7, width=0.5)
    ax.set_ylabel('Inference Time (ms/sample)', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Inference Speed', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.4f} ms', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add speedup annotation
    speedup = rf_inference_time / lgbm_inference_time
    ax.text(0.5, max(speeds)*0.8, f'LightGBM is {speedup:.2f}x faster',
            ha='center', fontsize=13, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comparison_inference_speed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_inference_speed.png")
    plt.close()
    
    # 4. Comparison - R² by SF
    fig, ax = plt.subplots(figsize=(12, 8))
    sf_combined = pd.merge(rf_sf_perf[['SF', 'R²']], lgbm_sf_perf[['SF', 'R²']], 
                           on='SF', suffixes=('_RF', '_LightGBM'))
    
    x = np.arange(len(sf_combined))
    width = 0.35
    
    ax.bar(x - width/2, sf_combined['R²_RF'], width, label='Random Forest', 
           color='blue', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, sf_combined['R²_LightGBM'], width, label='LightGBM', 
           color='green', edgecolor='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'SF{sf}' for sf in sf_combined['SF']], fontsize=13)
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: R² by Spreading Factor', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_r2_by_sf.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_r2_by_sf.png")
    plt.close()
    
    # 5. Comparison - Error Distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(residuals_rf, bins=40, alpha=0.6, label='Random Forest', 
            color='blue', edgecolor='black')
    ax.hist(residuals_lgbm, bins=40, alpha=0.6, label='LightGBM', 
            color='green', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=3, label='Zero Error')
    ax.set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Error Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_error_distribution.png")
    plt.close()
    
    # 6. Comparison - Improvement Summary
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_imp = ['green' if v > 0 else 'red' for v in improvements.values()]
    
    bars = ax.barh(list(improvements.keys()), list(improvements.values()), 
                   color=colors_imp, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('LightGBM vs Random Forest: Performance Improvement', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (metric, value) in enumerate(improvements.items()):
        ax.text(value, i, f'  {value:+.1f}%', 
                va='center', ha='left' if value > 0 else 'right',
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparison_improvement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_improvement.png")
    plt.close()

# ========== EXPORT RESULTS TO CSV ==========

print("\n💾 Exporting results to CSV...")

if lgbm_available:
    metrics_df = pd.DataFrame([rf_metrics, lgbm_metrics])
else:
    metrics_df = pd.DataFrame([rf_metrics])

metrics_df.to_csv('evaluation_metrics_overall.csv', index=False)
print("✓ Saved: evaluation_metrics_overall.csv")

rf_sf_perf.to_csv('evaluation_rf_by_sf.csv', index=False)
rf_dist_perf.to_csv('evaluation_rf_by_distance.csv', index=False)
print("✓ Saved: evaluation_rf_by_sf.csv")
print("✓ Saved: evaluation_rf_by_distance.csv")

if lgbm_available:
    lgbm_sf_perf.to_csv('evaluation_lgbm_by_sf.csv', index=False)
    lgbm_dist_perf.to_csv('evaluation_lgbm_by_distance.csv', index=False)
    print("✓ Saved: evaluation_lgbm_by_sf.csv")
    print("✓ Saved: evaluation_lgbm_by_distance.csv")

# ========== FINAL SUMMARY ==========

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)

print("\n📊 Summary:")
print(f"  Random Forest R²:     {rf_metrics['R²']:.4f}")
print(f"  Random Forest MAE:    {rf_metrics['MAE']:.2f}%")
print(f"  Random Forest RMSE:   {rf_metrics['RMSE']:.2f}%")

if lgbm_available:
    print(f"\n  LightGBM R²:          {lgbm_metrics['R²']:.4f}")
    print(f"  LightGBM MAE:         {lgbm_metrics['MAE']:.2f}%")
    print(f"  LightGBM RMSE:        {lgbm_metrics['RMSE']:.2f}%")
    
    print(f"\n💡 Performance Improvement:")
    print(f"  R² gain:              {improvements['R² Improvement']:+.2f}%")
    print(f"  MAE improvement:      {improvements['MAE Improvement']:+.2f}%")
    print(f"  RMSE improvement:     {improvements['RMSE Improvement']:+.2f}%")
    print(f"  Speed improvement:    {improvements['Speed Improvement']:+.2f}%")

print("\n📁 Generated Image Files:")
print("\n  Random Forest (6 images):")
print("    ✓ rf_prediction_scatter.png")
print("    ✓ rf_residual_plot.png")
print("    ✓ rf_error_distribution.png")
print("    ✓ rf_performance_by_sf.png")
print("    ✓ rf_mae_by_distance.png")
print("    ✓ rf_feature_importance.png")

if lgbm_available:
    print("\n  LightGBM (6 images):")
    print("    ✓ lgbm_prediction_scatter.png")
    print("    ✓ lgbm_residual_plot.png")
    print("    ✓ lgbm_error_distribution.png")
    print("    ✓ lgbm_performance_by_sf.png")
    print("    ✓ lgbm_mae_by_distance.png")
    print("    ✓ lgbm_feature_importance.png")
    
    print("\n  Comparison (6 images):")
    print("    ✓ comparison_predictions.png")
    print("    ✓ comparison_metrics.png")
    print("    ✓ comparison_inference_speed.png")
    print("    ✓ comparison_r2_by_sf.png")
    print("    ✓ comparison_error_distribution.png")
    print("    ✓ comparison_improvement.png")

print("\n📄 CSV Files:")
print("    ✓ evaluation_metrics_overall.csv")
print("    ✓ evaluation_rf_by_sf.csv")
print("    ✓ evaluation_rf_by_distance.csv")
if lgbm_available:
    print("    ✓ evaluation_lgbm_by_sf.csv")
    print("    ✓ evaluation_lgbm_by_distance.csv")

print("\n" + "="*70)
print(f"Total: {6 + (12 if lgbm_available else 0)} image files generated!")
print("="*70)