#!/usr/bin/env python3
"""
LoRaWAN ML-ADR Model Analysis & Visualization
Generate SEPARATE image files for each plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("="*70)
print("LoRaWAN ML-ADR Model Analysis - Individual Plots")
print("="*70)

# ========== LOAD DATA & MODEL ==========

print("\n📂 Loading data and model...")
df = pd.read_csv('dataset_complete_with_rssi_snr.csv')
model = joblib.load('lorawan_adr_model_b_energy.pkl')
params = joblib.load('path_loss_params_energy.pkl')

print(f"✓ Loaded {len(df)} records")
print(f"✓ Loaded model and parameters")

# Feature engineering
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
y_true = df['PDR']
y_pred = model.predict(X)

# Metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\n📊 Model Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE:      {mae:.2f}%")
print(f"  RMSE:     {rmse:.2f}%")

# ========== PLOT 1: FEATURE IMPORTANCE ==========

print("\n📈 [1/8] Generating Feature Importance plot...")
plt.figure(figsize=(10, 8))

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

colors = plt.cm.viridis(feature_importance['Importance'] / feature_importance['Importance'].max())
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors, edgecolor='black')
plt.xlabel('Importance', fontsize=13, fontweight='bold')
plt.ylabel('Feature', fontsize=13, fontweight='bold')
plt.title('Feature Importance (Random Forest Model)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)

# Add percentage labels
for feat, imp in zip(feature_importance['Feature'], feature_importance['Importance']):
    plt.text(imp, feat, f' {imp:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('01_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_feature_importance.png")

# ========== PLOT 2: TRUE vs PREDICTED PDR ==========

print("📈 [2/8] Generating True vs Predicted PDR plot...")
plt.figure(figsize=(10, 8))

# Sample for clarity
sample_size = min(5000, len(y_true))
indices = np.random.choice(len(y_true), sample_size, replace=False)

scatter = plt.scatter(y_true.iloc[indices], y_pred[indices], 
                     c=df['SF'].iloc[indices], cmap='viridis', 
                     alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True PDR (%)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted PDR (%)', fontsize=13, fontweight='bold')
plt.title(f'True vs Predicted PDR\nR²={r2:.4f}, MAE={mae:.2f}%, RMSE={rmse:.2f}%', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Spreading Factor', fontsize=12)

# Add error statistics
errors = y_pred - y_true
plt.text(0.05, 0.95, f'Mean Error: {errors.mean():.2f}%\nStd Error: {errors.std():.2f}%',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('02_true_vs_predicted_pdr.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_true_vs_predicted_pdr.png")

# ========== PLOT 3: PREDICTION ERROR DISTRIBUTION ==========

print("📈 [3/8] Generating Error Distribution plot...")
plt.figure(figsize=(10, 8))

errors = y_pred - y_true
plt.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Prediction Error (%) = Predicted - True PDR', fontsize=13, fontweight='bold')
plt.ylabel('Frequency', fontsize=13, fontweight='bold')
plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add statistics
plt.text(0.95, 0.95, f'Mean: {errors.mean():.2f}%\nMedian: {np.median(errors):.2f}%\nStd: {errors.std():.2f}%\nMin: {errors.min():.2f}%\nMax: {errors.max():.2f}%',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('03_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_error_distribution.png")

# ========== PLOT 4: ENERGY vs PDR SCATTER (TRADE-OFF) ==========

print("📈 [4/8] Generating Energy vs PDR Trade-off plot...")
plt.figure(figsize=(12, 8))

scatter = plt.scatter(df['EnergyPerPacket_mJ'], df['PDR'], 
                     c=df['SF'], cmap='RdYlGn_r', s=50, alpha=0.6, 
                     edgecolors='black', linewidth=0.5)
plt.xlabel('Energy per Packet (mJ)', fontsize=13, fontweight='bold')
plt.ylabel('PDR (%)', fontsize=13, fontweight='bold')
plt.title('Energy vs PDR Trade-off\n(Colored by Spreading Factor)', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Spreading Factor', fontsize=12)

# Add optimal region annotations
plt.axhline(80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='PDR Threshold (80%)')
plt.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Energy Target (1mJ)')
plt.legend(loc='lower right', fontsize=11)

# Add annotation for optimal region
plt.annotate('Optimal Region\n(High PDR, Low Energy)', 
            xy=(0.5, 90), xytext=(2, 95),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('04_energy_vs_pdr_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_energy_vs_pdr_tradeoff.png")

# ========== PLOT 5: ENERGY BY SF (BOX PLOT) ==========

print("📈 [5/8] Generating Energy by SF box plot...")
plt.figure(figsize=(12, 8))

sf_data = []
sf_labels = []
for sf in sorted(df['SF'].unique()):
    sf_data.append(df[df['SF'] == sf]['EnergyPerPacket_mJ'])
    sf_labels.append(f'SF{sf}')

bp = plt.boxplot(sf_data, labels=sf_labels, patch_artist=True, 
                showmeans=True, meanline=True, widths=0.6)

# Color boxes
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sf_data)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

plt.xlabel('Spreading Factor', fontsize=13, fontweight='bold')
plt.ylabel('Energy per Packet (mJ)', fontsize=13, fontweight='bold')
plt.title('Energy Consumption Distribution by Spreading Factor', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')

# Add mean values as text
for i, sf in enumerate(sorted(df['SF'].unique()), 1):
    mean_energy = df[df['SF'] == sf]['EnergyPerPacket_mJ'].mean()
    plt.text(i, mean_energy, f'{mean_energy:.2f} mJ', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('05_energy_by_sf_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 05_energy_by_sf_boxplot.png")

# ========== PLOT 6: PDR BY SF (BOX PLOT) ==========

print("📈 [6/8] Generating PDR by SF box plot...")
plt.figure(figsize=(12, 8))

pdr_data = []
for sf in sorted(df['SF'].unique()):
    pdr_data.append(df[df['SF'] == sf]['PDR'])

bp = plt.boxplot(pdr_data, labels=sf_labels, patch_artist=True, 
                showmeans=True, meanline=True, widths=0.6)

# Color boxes
colors = plt.cm.RdYlGn(np.linspace(0, 1, len(pdr_data)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

plt.xlabel('Spreading Factor', fontsize=13, fontweight='bold')
plt.ylabel('PDR (%)', fontsize=13, fontweight='bold')
plt.title('PDR Distribution by Spreading Factor', 
          fontsize=14, fontweight='bold', pad=20)
plt.axhline(80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (80%)')
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 105])

# Add mean values
for i, sf in enumerate(sorted(df['SF'].unique()), 1):
    mean_pdr = df[df['SF'] == sf]['PDR'].mean()
    plt.text(i, mean_pdr, f'{mean_pdr:.1f}%', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('06_pdr_by_sf_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 06_pdr_by_sf_boxplot.png")

# ========== PLOT 7: CORRELATION HEATMAP ==========

print("📈 [7/8] Generating Correlation Heatmap...")
plt.figure(figsize=(12, 10))

# Select key features for correlation
corr_features = ['distance', 'SF', 'txPower', 'RSSI', 'SNR', 
                'link_margin_dB', 'EnergyPerPacket_mJ', 'PDR', 'AvgToA_ms']
corr_df = df[corr_features].corr()

# Create heatmap
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)  # Mask upper triangle
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=1.5, cbar_kws={"shrink": 0.8},
           mask=mask, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap\n(Key Variables)', 
         fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 07_correlation_heatmap.png")

# ========== PLOT 8: BATTERY LIFETIME BY SF ==========

print("📈 [8/8] Generating Battery Lifetime plot...")
plt.figure(figsize=(12, 8))

battery_capacity_J = 21.6  # 2000mAh @ 3V
packets_per_day = 100

sf_list = []
battery_years = []
for sf in sorted(df['SF'].unique()):
    avg_energy_J = df[df['SF'] == sf]['EnergyPerPacket_J'].mean()
    years = (battery_capacity_J / avg_energy_J) / packets_per_day / 365
    sf_list.append(sf)
    battery_years.append(years)

colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(sf_list)))
bars = plt.bar([f'SF{sf}' for sf in sf_list], battery_years, 
              color=colors, alpha=0.8, edgecolor='black', linewidth=2)

plt.xlabel('Spreading Factor', fontsize=13, fontweight='bold')
plt.ylabel('Battery Lifetime (years)', fontsize=13, fontweight='bold')
plt.title('Estimated Battery Lifetime by Spreading Factor\n(100 packets/day, 2000mAh @ 3V)', 
         fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, years in zip(bars, battery_years):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{years:.2f} years', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

# Add annotation
plt.text(0.98, 0.98, f'Target: 5+ years\nfor IoT sensors',
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('08_battery_lifetime_by_sf.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 08_battery_lifetime_by_sf.png")

# ========== GENERATE COMPARISON TABLES ==========

print("\n📊 Generating Comparison Tables...")

# Table 1: Distance-based Optimal Configs
print("\n" + "="*80)
print("TABLE 1: Distance-Based Optimal Configuration Comparison")
print("="*80)

test_distances = [500, 1000, 2000, 3000, 4000, 5000]
comparison_data = []

for dist in test_distances:
    configs_at_dist = df[df['distance'].between(dist-100, dist+100)]
    
    if len(configs_at_dist) > 0:
        # ToA-optimized (min ToA)
        toa_opt = configs_at_dist.loc[configs_at_dist['AvgToA_ms'].idxmin()]
        
        # Energy-optimized (min Energy)
        energy_opt = configs_at_dist.loc[configs_at_dist['EnergyPerPacket_mJ'].idxmin()]
        
        # Energy saving
        energy_saving = ((toa_opt['EnergyPerPacket_mJ'] - energy_opt['EnergyPerPacket_mJ']) / 
                        toa_opt['EnergyPerPacket_mJ'] * 100)
        
        comparison_data.append({
            'Distance_m': dist,
            'ToA_Opt_SF': int(toa_opt['SF']),
            'ToA_Opt_Power_dBm': int(toa_opt['txPower']),
            'ToA_ms': round(toa_opt['AvgToA_ms'], 1),
            'ToA_Energy_mJ': round(toa_opt['EnergyPerPacket_mJ'], 3),
            'ToA_PDR_%': round(toa_opt['PDR'], 1),
            'Energy_Opt_SF': int(energy_opt['SF']),
            'Energy_Opt_Power_dBm': int(energy_opt['txPower']),
            'Energy_ToA_ms': round(energy_opt['AvgToA_ms'], 1),
            'Energy_mJ': round(energy_opt['EnergyPerPacket_mJ'], 3),
            'Energy_PDR_%': round(energy_opt['PDR'], 1),
            'Energy_Saving_%': round(energy_saving, 1)
        })

comp_df = pd.DataFrame(comparison_data)
print(comp_df.to_string(index=False))
comp_df.to_csv('table1_comparison_toa_vs_energy.csv', index=False)
print("\n✅ Saved: table1_comparison_toa_vs_energy.csv")

# Table 2: SF Summary
print("\n" + "="*80)
print("TABLE 2: Spreading Factor Summary Statistics")
print("="*80)

sf_summary_data = []
for sf in sorted(df['SF'].unique()):
    sf_df = df[df['SF'] == sf]
    
    avg_energy_J = sf_df['EnergyPerPacket_J'].mean()
    battery_years = (battery_capacity_J / avg_energy_J) / packets_per_day / 365
    
    sf_summary_data.append({
        'SF': int(sf),
        'PDR_Mean_%': round(sf_df['PDR'].mean(), 1),
        'PDR_Std_%': round(sf_df['PDR'].std(), 1),
        'PDR_Min_%': round(sf_df['PDR'].min(), 1),
        'PDR_Max_%': round(sf_df['PDR'].max(), 1),
        'Energy_Mean_mJ': round(sf_df['EnergyPerPacket_mJ'].mean(), 3),
        'Energy_Std_mJ': round(sf_df['EnergyPerPacket_mJ'].std(), 3),
        'ToA_Mean_ms': round(sf_df['AvgToA_ms'].mean(), 1),
        'Distance_Mean_m': round(sf_df['distance'].mean(), 1),
        'Sample_Count': len(sf_df),
        'Battery_Years': round(battery_years, 2)
    })

sf_summary = pd.DataFrame(sf_summary_data)
print(sf_summary.to_string(index=False))
sf_summary.to_csv('table2_sf_summary_statistics.csv', index=False)
print("\n✅ Saved: table2_sf_summary_statistics.csv")

# Table 3: Energy Savings Analysis
print("\n" + "="*80)
print("TABLE 3: Energy Savings Potential Analysis")
print("="*80)

energy_savings_summary = []
for dist_range in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000)]:
    subset = df[df['distance'].between(dist_range[0], dist_range[1])]
    
    if len(subset) > 0:
        min_toa_config = subset.loc[subset['AvgToA_ms'].idxmin()]
        min_energy_config = subset.loc[subset['EnergyPerPacket_mJ'].idxmin()]
        
        savings = ((min_toa_config['EnergyPerPacket_mJ'] - min_energy_config['EnergyPerPacket_mJ']) / 
                  min_toa_config['EnergyPerPacket_mJ'] * 100)
        
        energy_savings_summary.append({
            'Distance_Range_m': f"{dist_range[0]}-{dist_range[1]}",
            'Sample_Count': len(subset),
            'ToA_Opt_SF': int(min_toa_config['SF']),
            'ToA_Opt_Energy_mJ': round(min_toa_config['EnergyPerPacket_mJ'], 3),
            'Energy_Opt_SF': int(min_energy_config['SF']),
            'Energy_Opt_mJ': round(min_energy_config['EnergyPerPacket_mJ'], 3),
            'Saving_mJ': round(min_toa_config['EnergyPerPacket_mJ'] - min_energy_config['EnergyPerPacket_mJ'], 3),
            'Saving_%': round(savings, 1),
            'ToA_Increase_ms': round(min_energy_config['AvgToA_ms'] - min_toa_config['AvgToA_ms'], 1)
        })

savings_df = pd.DataFrame(energy_savings_summary)
print(savings_df.to_string(index=False))
savings_df.to_csv('table3_energy_savings_analysis.csv', index=False)
print("\n✅ Saved: table3_energy_savings_analysis.csv")

# ========== SUMMARY ==========

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS AND TABLES GENERATED SUCCESSFULLY!")
print("="*80)
print("\n📊 Generated Image Files (8 plots):")
print("  1. 01_feature_importance.png           - Feature importance ranking")
print("  2. 02_true_vs_predicted_pdr.png        - Model accuracy visualization")
print("  3. 03_error_distribution.png           - Prediction error analysis")
print("  4. 04_energy_vs_pdr_tradeoff.png       - Multi-objective trade-off")
print("  5. 05_energy_by_sf_boxplot.png         - Energy distribution by SF")
print("  6. 06_pdr_by_sf_boxplot.png            - PDR distribution by SF")
print("  7. 07_correlation_heatmap.png          - Feature correlations")
print("  8. 08_battery_lifetime_by_sf.png       - Battery lifetime estimation")

print("\n📋 Generated CSV Tables (3 tables):")
print("  1. table1_comparison_toa_vs_energy.csv - Distance-based comparison")
print("  2. table2_sf_summary_statistics.csv    - SF statistics")
print("  3. table3_energy_savings_analysis.csv  - Energy savings potential")

print("\n💡 All files saved in current directory!")
print("="*80)