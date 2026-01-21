#!/usr/bin/env python3
"""
LoRaWAN ML-Based ADR - ENERGY-AWARE VERSION
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
warnings.filterwarnings('ignore')

print("="*70)
print("LoRaWAN ML-Based ADR - Energy-Aware Configuration")
print("Multi-Objective: PDR + Energy Efficiency")
print("="*70)

# ========== LOAD DATASET ==========

df = pd.read_csv('dataset_complete_with_rssi_snr.csv')
print(f"\n✓ Loaded {len(df)} records")
print(f"✓ Unique experiments: {df['experiment'].nunique()}")
print(f"✓ Unique nodes: {df['node_id'].nunique()}")

# Show sample with energy data
print("\n📊 Sample data:")
print(df[['node_id', 'distance', 'dr', 'SF', 'txPower', 'RSSI', 'PDR', 
          'AvgToA_ms', 'EnergyPerPacket_J']].head())

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

# 5. ⭐ ENERGY FEATURES (NEW!)
# Energy per packet in millijoules
df['EnergyPerPacket_mJ'] = df['EnergyPerPacket_J'] * 1000

# Calculate energy from TxPower and ToA (for validation)
def calculate_energy_from_params(txPower_dBm, toa_ms):
    """Calculate energy from TX power and time-on-air"""
    txPower_mW = 10 ** (txPower_dBm / 10.0)
    toa_s = toa_ms / 1000.0
    return txPower_mW * toa_s

df['Calculated_Energy_mJ'] = df.apply(
    lambda row: calculate_energy_from_params(row['txPower'], row['AvgToA_ms']), 
    axis=1
)

# Verify energy calculation
print(f"\n✓ Energy Validation:")
print(f"  Dataset Energy (mean): {df['EnergyPerPacket_mJ'].mean():.3f} mJ")
print(f"  Calculated Energy (mean): {df['Calculated_Energy_mJ'].mean():.3f} mJ")
print(f"  Correlation: {df['EnergyPerPacket_mJ'].corr(df['Calculated_Energy_mJ']):.4f}")

# 6. ⭐ ENERGY-BASED MULTI-OBJECTIVE SCORE
df['pdr_normalized'] = df['PDR'] / 100.0

# Energy efficiency: lower is better, so we normalize inversely
max_energy = df['EnergyPerPacket_mJ'].max()
df['energy_efficiency'] = 1 - (df['EnergyPerPacket_mJ'] / max_energy)

# Multi-objective score: 60% PDR, 40% Energy Efficiency
df['performance_score'] = 0.6 * df['pdr_normalized'] + 0.4 * df['energy_efficiency']

print("✓ Created energy-aware features")

# ========== ENERGY ANALYSIS ==========

print("\n" + "="*70)
print("Energy Consumption Analysis")
print("="*70)

# Energy per SF
print("\nEnergy Consumption by Spreading Factor:")
energy_by_sf = df.groupby('SF').agg({
    'EnergyPerPacket_mJ': ['mean', 'std', 'min', 'max'],
    'AvgToA_ms': 'mean',
    'txPower': 'mean'
}).round(3)
print(energy_by_sf)

# Energy vs PDR correlation
print(f"\n📊 Energy vs PDR correlation: {df['EnergyPerPacket_mJ'].corr(df['PDR']):.4f}")

# Battery lifetime estimation
print("\n🔋 Battery Lifetime Estimation (2000mAh @ 3V = 21,600 J):")
battery_capacity_J = 21600
packets_per_day = 100

for sf in sorted(df['SF'].unique()):
    avg_energy_J = df[df['SF'] == sf]['EnergyPerPacket_J'].mean()
    packets_lifetime = battery_capacity_J / avg_energy_J
    days_lifetime = packets_lifetime / packets_per_day
    years_lifetime = days_lifetime / 365
    
    print(f"  SF{sf}: {avg_energy_J*1000:.2f} mJ/pkt → "
          f"{packets_lifetime:.0f} pkts → {years_lifetime:.1f} years")

# ========== FIT PATH LOSS MODEL ==========

print("\n" + "="*70)
print("Fitting Path Loss Model from NS-3 Data...")
print("="*70)

df['path_loss_dB'] = df['txPower'] - df['RSSI']

from scipy.optimize import curve_fit

def path_loss_model(distance, PL0, n):
    """Log-distance path loss model"""
    d0 = 1.0
    return PL0 + 10 * n * np.log10(distance / d0)

distances = df['distance'].values
path_losses = df['path_loss_dB'].values

params, covariance = curve_fit(path_loss_model, distances, path_losses, p0=[7.7, 3.76])
PL0_fitted, n_fitted = params

print(f"✓ Fitted Path Loss Model:")
print(f"  PL0: {PL0_fitted:.2f} dB")
print(f"  n:   {n_fitted:.2f}")

predicted_pl = path_loss_model(distances, PL0_fitted, n_fitted)
pl_rmse = np.sqrt(mean_squared_error(path_losses, predicted_pl))
print(f"  RMSE: {pl_rmse:.2f} dB")

# ========== NS-3 PARAMETERS ==========

NOISE_FLOOR_DBM = -117.03

SENSITIVITY_MAP = {
    0: -142.5,  # DR0 = SF12
    1: -140.0,  # DR1 = SF11
    2: -137.5,  # DR2 = SF10
    3: -135.0,  # DR3 = SF9
    4: -132.5,  # DR4 = SF8
    5: -130.0   # DR5 = SF7
}

TOA_LOOKUP = {
    0: 991.232,  # DR0 = SF12
    1: 495.616,  # DR1 = SF11
    2: 247.808,  # DR2 = SF10
    3: 128.0,    # DR3 = SF9
    4: 66.048,   # DR4 = SF8
    5: 34.048    # DR5 = SF7
}

# ⭐ ENERGY LOOKUP (calculated from TxPower and ToA)
def build_energy_lookup():
    """Build energy consumption lookup table for all DR and TxPower combinations"""
    energy_lookup = {}
    
    for dr in [0, 1, 2, 3, 4, 5]:
        energy_lookup[dr] = {}
        for power in [6, 8, 10, 12, 14]:
            toa_ms = TOA_LOOKUP[dr]
            energy_mJ = calculate_energy_from_params(power, toa_ms)
            energy_lookup[dr][power] = energy_mJ
    
    return energy_lookup

ENERGY_LOOKUP = build_energy_lookup()

print("\n✓ Energy Lookup Table (mJ per packet):")
print("     Power (dBm):", "    6     8    10    12    14")
for dr in [0, 1, 2, 3, 4, 5]:
    sf = 12 - dr
    energies = [ENERGY_LOOKUP[dr][p] for p in [6, 8, 10, 12, 14]]
    print(f"  SF{sf}: {' '.join([f'{e:6.2f}' for e in energies])}")

# ========== HELPER FUNCTIONS ==========

def calculate_link_quality(distance, txPower, dr):
    """Calculate RSSI, SNR, Sensitivity, Margins"""
    path_loss = path_loss_model(distance, PL0_fitted, n_fitted)
    rssi = txPower - path_loss
    snr = rssi - NOISE_FLOOR_DBM
    sensitivity = SENSITIVITY_MAP.get(dr, -130.0)
    link_margin = rssi - sensitivity
    snr_margin = snr - (-7.5)
    is_above = 1 if rssi > sensitivity else 0
    
    return rssi, snr, sensitivity, link_margin, snr_margin, is_above

def calculate_energy_consumption(txPower_dBm, dr):
    """Get energy consumption from lookup table"""
    return ENERGY_LOOKUP[dr][txPower_dBm]

# ========== TRAIN MODEL B WITH ENERGY AWARENESS ==========

print("\n" + "="*70)
print("Training Model B with Energy Features...")
print("="*70)

features_model_b = [
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
    'EnergyPerPacket_mJ'  # ⭐ ADD ENERGY FEATURE
]

X_b = df[features_model_b]
y_pdr = df['PDR']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_b, y_pdr, test_size=0.2, random_state=42
)

model_b = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)

model_b.fit(X_train_b, y_train_b)
y_pred_b = model_b.predict(X_test_b)

r2_b = r2_score(y_test_b, y_pred_b)
mae_b = mean_absolute_error(y_test_b, y_pred_b)
rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))

print(f"\n🟢 Model B (Energy-Aware):")
print(f"  R² Score: {r2_b:.4f}")
print(f"  MAE:      {mae_b:.2f}%")
print(f"  RMSE:     {rmse_b:.2f}%")

# Feature Importance
feature_importance_b = pd.DataFrame({
    'Feature': features_model_b,
    'Importance': model_b.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("Feature Importance:")
print("="*70)
print(feature_importance_b.head(10).to_string(index=False))

# ========== VISUALIZATION ==========

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: PDR Prediction
axes[0, 0].scatter(y_test_b, y_pred_b, alpha=0.6, s=50, edgecolors='black')
axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
axes[0, 0].set_xlabel('True PDR (%)', fontsize=11)
axes[0, 0].set_ylabel('Predicted PDR (%)', fontsize=11)
axes[0, 0].set_title(f'PDR Prediction (R²={r2_b:.3f})', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Feature Importance
top_features = feature_importance_b.head(10)
axes[0, 1].barh(top_features['Feature'], top_features['Importance'], 
                color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('Importance', fontsize=11)
axes[0, 1].set_title('Top 10 Features', fontweight='bold')
axes[0, 1].invert_yaxis()

# Plot 3: Energy vs PDR
axes[0, 2].scatter(df['EnergyPerPacket_mJ'], df['PDR'], 
                   c=df['SF'], cmap='viridis', s=30, alpha=0.6)
cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], label='SF')
axes[0, 2].set_xlabel('Energy per Packet (mJ)', fontsize=11)
axes[0, 2].set_ylabel('PDR (%)', fontsize=11)
axes[0, 2].set_title('Energy vs PDR by SF', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Energy by SF
sf_energy = df.groupby('SF')['EnergyPerPacket_mJ'].mean()
axes[1, 0].bar(sf_energy.index, sf_energy.values, 
               color='orange', edgecolor='black', alpha=0.8)
axes[1, 0].set_xlabel('Spreading Factor', fontsize=11)
axes[1, 0].set_ylabel('Average Energy (mJ)', fontsize=11)
axes[1, 0].set_title('Energy Consumption by SF', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, (sf, energy) in enumerate(sf_energy.items()):
    axes[1, 0].text(i, energy, f'{energy:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 5: Battery Lifetime by SF
battery_capacity_J = 21.6  # 2000mAh @ 3V
packets_per_day = 100
lifetime_years = []
for sf in sorted(df['SF'].unique()):
    avg_energy_J = df[df['SF'] == sf]['EnergyPerPacket_J'].mean()
    years = (battery_capacity_J / avg_energy_J) / packets_per_day / 365
    lifetime_years.append(years)

axes[1, 1].bar(sorted(df['SF'].unique()), lifetime_years, 
               color='green', edgecolor='black', alpha=0.8)
axes[1, 1].set_xlabel('Spreading Factor', fontsize=11)
axes[1, 1].set_ylabel('Battery Lifetime (years)', fontsize=11)
axes[1, 1].set_title('Estimated Battery Life\n(100 pkt/day, 2000mAh)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (sf, years) in enumerate(zip(sorted(df['SF'].unique()), lifetime_years)):
    axes[1, 1].text(i, years, f'{years:.1f}y', ha='center', va='bottom', fontweight='bold')

# Plot 6: PDR vs Link Margin (colored by energy)
test_b_df = X_test_b.copy()
test_b_df['True_PDR'] = y_test_b.values
scatter = axes[1, 2].scatter(test_b_df['link_margin_dB'], test_b_df['True_PDR'], 
                             c=test_b_df['EnergyPerPacket_mJ'], cmap='RdYlGn_r', 
                             s=40, alpha=0.6, edgecolors='black')
plt.colorbar(scatter, ax=axes[1, 2], label='Energy (mJ)')
axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Link Margin (dB)', fontsize=11)
axes[1, 2].set_ylabel('PDR (%)', fontsize=11)
axes[1, 2].set_title('PDR vs Link Margin\n(colored by energy)', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_energy_aware_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: ml_energy_aware_evaluation.png")

# ========== ENERGY-AWARE RECOMMENDATION FUNCTION ==========

def recommend_best_config_energy(distance, model, top_n=5, 
                                 pdr_threshold=80, safety_margin_dB=2,
                                 weight_pdr=0.6, weight_energy=0.4):
    """
    Recommend best configuration with ENERGY OPTIMIZATION
    
    Multi-objective: PDR (reliability) + Energy Efficiency
    """
    print(f"\n{'='*70}")
    print(f"Energy-Aware Config for Distance: {distance:.0f}m")
    print(f"Weights: PDR={weight_pdr:.0%}, Energy={weight_energy:.0%}")
    print(f"{'='*70}")
    
    recommendations = []
    
    # Calculate max energy for normalization
    max_energy_mJ = max([ENERGY_LOOKUP[dr][14] for dr in range(6)])
    
    for dr in [0, 1, 2, 3, 4, 5]:
        for freq in [868.1, 868.3, 868.5]:
            for power in [6, 8, 10, 12, 14]:
                
                # Calculate link quality
                rssi, snr, sensitivity, link_margin, snr_margin, is_above = \
                    calculate_link_quality(distance, power, dr)
                
                # Safety filter
                if link_margin < safety_margin_dB:
                    continue
                
                # Get energy consumption
                energy_mJ = ENERGY_LOOKUP[dr][power]
                
                # Prepare features for prediction
                distance_norm = distance / df['distance'].max()
                radial = distance
                power_dist_ratio = power / (distance + 1)
                
                X_config = pd.DataFrame([{
                    'distance': distance,
                    'distance_normalized': distance_norm,
                    'radial_distance': radial,
                    'dr': dr,
                    'freq': freq,
                    'txPower': power,
                    'RSSI': rssi,
                    'SNR': snr,
                    'Sensitivity': sensitivity,
                    'link_margin_dB': link_margin,
                    'snr_margin_dB': snr_margin,
                    'is_above_sensitivity': is_above,
                    'power_distance_ratio': power_dist_ratio,
                    'EnergyPerPacket_mJ': energy_mJ
                }])
                
                # Predict PDR
                pred_pdr = model.predict(X_config)[0]
                pred_pdr = max(0, min(100, pred_pdr))
                
                # Calculate multi-objective score
                if pred_pdr >= pdr_threshold:
                    pdr_score = pred_pdr / 100.0
                    energy_score = 1 - (energy_mJ / max_energy_mJ)
                    
                    # ⭐ ENERGY-BASED SCORE
                    score = weight_pdr * pdr_score + weight_energy * energy_score
                else:
                    score = 0
                
                sf = 12 - dr
                
                # Battery lifetime (100 pkt/day, 2000mAh @ 3V)
                battery_J = 21.6
                energy_J = energy_mJ / 1000
                lifetime_days = battery_J / (energy_J * 100)
                lifetime_years = lifetime_days / 365
                
                recommendations.append({
                    'DR': dr,
                    'SF': sf,
                    'Freq_MHz': freq,
                    'TxPower_dBm': power,
                    'Predicted_PDR_%': round(pred_pdr, 1),
                    'Energy_mJ': round(energy_mJ, 3),
                    'Battery_Years': round(lifetime_years, 2),
                    'RSSI_dBm': round(rssi, 1),
                    'LinkMargin_dB': round(link_margin, 1),
                    'Score': round(score, 4)
                })
    
    if len(recommendations) == 0:
        print(f"\n⚠️  No safe configs found!")
        if safety_margin_dB > 1:
            return recommend_best_config_energy(distance, model, top_n, 
                                               pdr_threshold, safety_margin_dB=1,
                                               weight_pdr=weight_pdr, weight_energy=weight_energy)
        return pd.DataFrame()
    
    rec_df = pd.DataFrame(recommendations).sort_values('Score', ascending=False)
    
    print(f"\n✓ Found {len(rec_df)} safe configurations\n")
    print(f"Top {min(top_n, len(rec_df))} Energy-Optimized Configs:")
    print("="*70)
    
    display_cols = ['SF', 'Freq_MHz', 'TxPower_dBm', 'Predicted_PDR_%', 
                    'Energy_mJ', 'Battery_Years', 'LinkMargin_dB']
    print(rec_df[display_cols].head(top_n).to_string(index=False))
    
    # Show best config
    best = rec_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"🎯 BEST ENERGY-EFFICIENT CONFIG:")
    print(f"{'='*70}")
    print(f"  SF{best['SF']} | {best['Freq_MHz']} MHz | {best['TxPower_dBm']} dBm")
    print(f"  Expected PDR:      {best['Predicted_PDR_%']:.1f}%")
    print(f"  Energy per packet: {best['Energy_mJ']:.3f} mJ")
    print(f"  Battery lifetime:  {best['Battery_Years']:.1f} years")
    print(f"  Link margin:       {best['LinkMargin_dB']:.1f} dB")
    print(f"{'='*70}")
    
    return rec_df.head(top_n)

# ========== TEST ENERGY-AWARE RECOMMENDATIONS ==========

print("\n" + "="*70)
print("Energy-Aware Configuration Recommendations")
print("="*70)

test_distances = [1000, 2500, 4000, 5500]

for dist in test_distances:
    recommend_best_config_energy(dist, model_b, top_n=3, 
                                 pdr_threshold=80, safety_margin_dB=2,
                                 weight_pdr=0.6, weight_energy=0.4)

# ========== COMPARE: ToA vs Energy OPTIMIZATION ==========

print("\n" + "="*70)
print("COMPARISON: ToA-based vs Energy-based Optimization")
print("="*70)

def get_toa_optimized_config(distance):
    """Get config optimized for ToA"""
    configs = []
    for dr in [0, 1, 2, 3, 4, 5]:
        for power in [6, 8, 10, 12, 14]:
            rssi, _, _, link_margin, _, _ = calculate_link_quality(distance, power, dr)
            if link_margin >= 2:
                toa = TOA_LOOKUP[dr]
                configs.append({'DR': dr, 'SF': 12-dr, 'Power': power, 
                              'ToA_ms': toa, 'Energy_mJ': ENERGY_LOOKUP[dr][power],
                              'RSSI': rssi, 'Margin': link_margin})
    
    if configs:
        return min(configs, key=lambda x: x['ToA_ms'])
    return None

def get_energy_optimized_config(distance):
    """Get config optimized for Energy"""
    configs = []
    for dr in [0, 1, 2, 3, 4, 5]:
        for power in [6, 8, 10, 12, 14]:
            rssi, _, _, link_margin, _, _ = calculate_link_quality(distance, power, dr)
            if link_margin >= 2:
                energy = ENERGY_LOOKUP[dr][power]
                configs.append({'DR': dr, 'SF': 12-dr, 'Power': power, 
                              'ToA_ms': TOA_LOOKUP[dr], 'Energy_mJ': energy,
                              'RSSI': rssi, 'Margin': link_margin})
    
    if configs:
        return min(configs, key=lambda x: x['Energy_mJ'])
    return None

print("\nComparison at different distances:")
print("="*70)

for dist in [1000, 3000, 5000]:
    toa_config = get_toa_optimized_config(dist)
    energy_config = get_energy_optimized_config(dist)
    
    print(f"\nDistance: {dist}m")
    if toa_config:
        print(f"  ToA-Optimized:    SF{toa_config['SF']}, {toa_config['Power']}dBm")
        print(f"    → ToA: {toa_config['ToA_ms']:.1f} ms, Energy: {toa_config['Energy_mJ']:.2f} mJ")
    if energy_config:
        print(f"  Energy-Optimized: SF{energy_config['SF']}, {energy_config['Power']}dBm")
        print(f"    → ToA: {energy_config['ToA_ms']:.1f} ms, Energy: {energy_config['Energy_mJ']:.2f} mJ")
    
    if toa_config and energy_config:
        energy_saving = ((toa_config['Energy_mJ'] - energy_config['Energy_mJ']) / 
                        toa_config['Energy_mJ'] * 100)
        print(f"  💡 Energy saving: {energy_saving:.1f}%")

# ========== SAVE MODELS ==========

import joblib
joblib.dump(model_b, 'lorawan_adr_model_b_energy.pkl')

path_loss_params = {
    'PL0': PL0_fitted,
    'n': n_fitted,
    'noise_floor_dBm': NOISE_FLOOR_DBM,
    'sensitivity_map': SENSITIVITY_MAP,
    'toa_lookup': TOA_LOOKUP,
    'energy_lookup': ENERGY_LOOKUP
}
joblib.dump(path_loss_params, 'path_loss_params_energy.pkl')

print("\n✓ Models saved:")
print("  - lorawan_adr_model_b_energy.pkl (energy-aware model)")
print("  - path_loss_params_energy.pkl (parameters with energy lookup)")

print("\n" + "="*70)
print("✅ Energy-Aware Training Complete!")
print("="*70)
print(f"Model Performance:")
print(f"  R²:   {r2_b:.4f}")
print(f"  MAE:  {mae_b:.2f}%")
print("\nTop 3 Most Important Features:")
for _, row in feature_importance_b.head(3).iterrows():
    print(f"  - {row['Feature']}: {row['Importance']:.4f}")
print("="*70)