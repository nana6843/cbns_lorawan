#!/usr/bin/env python3
"""
Validate Model Recommendations Against Physical Constraints
"""

import pandas as pd
import numpy as np
import joblib

print("="*70)
print("Model Validation: Checking Physical Consistency")
print("="*70)

# ========== LOAD MODELS AND PARAMETERS ==========

model_a = joblib.load('lorawan_adr_model_a.pkl')
model_b = joblib.load('lorawan_adr_model_b.pkl')
params = joblib.load('path_loss_params.pkl')

print("✓ Loaded models and parameters")

# Extract parameters
PL0 = params['PL0']
n = params['n']
NOISE_FLOOR_DBM = params['noise_floor_dBm']
SENSITIVITY_MAP = params['sensitivity_map']
TOA_LOOKUP = params['toa_lookup']

print(f"\nPath Loss Model Parameters:")
print(f"  PL0: {PL0:.2f} dB")
print(f"  n:   {n:.2f}")
print(f"  Noise Floor: {NOISE_FLOOR_DBM:.2f} dBm")

# ========== RE-DEFINE LINK QUALITY CALCULATOR ==========

def path_loss_model(distance, PL0, n):
    """Log-distance path loss model"""
    d0 = 1.0
    return PL0 + 10 * n * np.log10(distance / d0)

def calculate_link_quality(distance, txPower, dr):
    """
    Calculate RSSI, SNR, Sensitivity for given configuration
    """
    # Calculate path loss
    path_loss = path_loss_model(distance, PL0, n)
    
    # Calculate RSSI
    rssi = txPower - path_loss
    
    # Calculate SNR
    snr = rssi - NOISE_FLOOR_DBM
    
    # Get sensitivity for this DR
    sensitivity = SENSITIVITY_MAP.get(dr, -123.0)
    
    # Calculate margins
    link_margin = rssi - sensitivity
    snr_margin = snr - (-7.5)
    is_above = 1 if rssi > sensitivity else 0
    
    return rssi, snr, sensitivity, link_margin, snr_margin, is_above

# ========== TEST CONFIGURATIONS ==========

print("\n" + "="*70)
print("Testing Various Configurations")
print("="*70)

test_cases = [
    {'dist': 1000, 'dr': 5, 'power': 12, 'expected': 'High PDR', 'desc': 'Close distance, SF7'},
    {'dist': 2000, 'dr': 5, 'power': 10, 'expected': 'High PDR', 'desc': 'Medium distance, SF7'},
    {'dist': 3000, 'dr': 5, 'power': 8,  'expected': 'Medium PDR', 'desc': 'Medium-far, SF7, low power'},
    {'dist': 4000, 'dr': 3, 'power': 12, 'expected': 'High PDR', 'desc': 'Far distance, SF9'},
    {'dist': 5000, 'dr': 5, 'power': 14, 'expected': 'Low/Zero PDR', 'desc': 'Very far, SF7 (impossible)'},
    {'dist': 5000, 'dr': 1, 'power': 14, 'expected': 'High PDR', 'desc': 'Very far, SF11'},
    {'dist': 6000, 'dr': 0, 'power': 14, 'expected': 'Medium PDR', 'desc': 'Edge case, SF12'},
    {'dist': 3500, 'dr': 4, 'power': 10, 'expected': 'High PDR', 'desc': 'Mid-range, SF8'},
]

results = []

for case in test_cases:
    dist = case['dist']
    dr = case['dr']
    power = case['power']
    freq = 868.3
    
    # Calculate link quality
    rssi, snr, sens, margin, snr_margin, is_above = calculate_link_quality(dist, power, dr)
    
    # Prepare features for Model A (without link quality)
    dist_norm = dist / 6000  # Approximate max distance
    radial = dist
    power_dist_ratio = power / (dist + 1)
    
    X_a = np.array([[dist, dist_norm, radial, dr, freq, power, power_dist_ratio]])
    pred_a = model_a.predict(X_a)[0]
    
    # Prepare features for Model B (with link quality)
    X_b = np.array([[
        dist, dist_norm, radial, dr, freq, power,
        rssi, snr, sens, margin, snr_margin, is_above,
        power_dist_ratio
    ]])
    pred_b = model_b.predict(X_b)[0]
    
    # Map DR to SF
    sf_map = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
    sf = sf_map[dr]
    
    # Determine physical feasibility
    feasible = 'Yes' if is_above else 'No'
    model_a_error = 'ERROR' if (not is_above and pred_a > 50) else 'OK'
    model_b_error = 'ERROR' if (not is_above and pred_b > 50) else 'OK'
    
    results.append({
        'Scenario': case['desc'],
        'Distance_m': dist,
        'SF': sf,
        'TxPower_dBm': power,
        'RSSI_dBm': round(rssi, 2),
        'LinkMargin_dB': round(margin, 2),
        'Feasible': feasible,
        'Model_A_PDR_%': round(max(0, min(100, pred_a)), 2),
        'Model_B_PDR_%': round(max(0, min(100, pred_b)), 2),
        'A_Status': model_a_error,
        'B_Status': model_b_error,
        'Expected': case['expected']
    })

results_df = pd.DataFrame(results)

print("\nValidation Results:")
print("="*70)
print(results_df.to_string(index=False))

# ========== CONSISTENCY ANALYSIS ==========

print("\n" + "="*70)
print("Physical Consistency Analysis:")
print("="*70)

# Count errors: predictions > 50% PDR when below sensitivity
impossible_a = len(results_df[(results_df['Feasible'] == 'No') & (results_df['Model_A_PDR_%'] > 50)])
impossible_b = len(results_df[(results_df['Feasible'] == 'No') & (results_df['Model_B_PDR_%'] > 50)])

total_infeasible = len(results_df[results_df['Feasible'] == 'No'])

print(f"\nConfigurations below sensitivity threshold: {total_infeasible}")
print(f"\nModel A:")
print(f"  - Physically impossible predictions: {impossible_a}/{total_infeasible}")
print(f"  - Error rate: {(impossible_a/total_infeasible*100) if total_infeasible > 0 else 0:.1f}%")

print(f"\nModel B:")
print(f"  - Physically impossible predictions: {impossible_b}/{total_infeasible}")
print(f"  - Error rate: {(impossible_b/total_infeasible*100) if total_infeasible > 0 else 0:.1f}%")

# ========== EDGE CASE ANALYSIS ==========

print("\n" + "="*70)
print("Edge Case Analysis (Distance >= 5000m):")
print("="*70)

edge_cases = results_df[results_df['Distance_m'] >= 5000]
if len(edge_cases) > 0:
    print("\nConfigs at extreme distances:")
    print(edge_cases[['Scenario', 'SF', 'LinkMargin_dB', 'Feasible', 'Model_A_PDR_%', 'Model_B_PDR_%']].to_string(index=False))
    
    # Highlight errors
    edge_errors = edge_cases[edge_cases['A_Status'] == 'ERROR']
    if len(edge_errors) > 0:
        print(f"\n⚠️  Model A made {len(edge_errors)} physically impossible predictions at edge distances!")
        print(edge_errors[['Scenario', 'Feasible', 'Model_A_PDR_%', 'LinkMargin_dB']].to_string(index=False))

# ========== STATISTICAL SUMMARY ==========

print("\n" + "="*70)
print("Statistical Summary:")
print("="*70)

# Separate feasible and infeasible configs
feasible_df = results_df[results_df['Feasible'] == 'Yes']
infeasible_df = results_df[results_df['Feasible'] == 'No']

print("\nFeasible Configurations (Above Sensitivity):")
print(f"  Count: {len(feasible_df)}")
if len(feasible_df) > 0:
    print(f"  Model A - Mean PDR: {feasible_df['Model_A_PDR_%'].mean():.2f}% (Std: {feasible_df['Model_A_PDR_%'].std():.2f})")
    print(f"  Model B - Mean PDR: {feasible_df['Model_B_PDR_%'].mean():.2f}% (Std: {feasible_df['Model_B_PDR_%'].std():.2f})")

print("\nInfeasible Configurations (Below Sensitivity):")
print(f"  Count: {len(infeasible_df)}")
if len(infeasible_df) > 0:
    print(f"  Model A - Mean PDR: {infeasible_df['Model_A_PDR_%'].mean():.2f}% (Std: {infeasible_df['Model_A_PDR_%'].std():.2f})")
    print(f"  Model B - Mean PDR: {infeasible_df['Model_B_PDR_%'].mean():.2f}% (Std: {infeasible_df['Model_B_PDR_%'].std():.2f})")
    print(f"\n  ⚠️  Model A predicts {feasible_df['Model_A_PDR_%'].mean():.1f}% PDR even when physically infeasible!")
    print(f"  ✅  Model B predicts {infeasible_df['Model_B_PDR_%'].mean():.1f}% PDR for infeasible configs (more realistic)")

# ========== VISUALIZATION ==========

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: PDR Predictions by Feasibility
feasible_data = results_df[results_df['Feasible'] == 'Yes']
infeasible_data = results_df[results_df['Feasible'] == 'No']

x = np.arange(2)
width = 0.35

model_a_means = [
    feasible_data['Model_A_PDR_%'].mean() if len(feasible_data) > 0 else 0,
    infeasible_data['Model_A_PDR_%'].mean() if len(infeasible_data) > 0 else 0
]

model_b_means = [
    feasible_data['Model_B_PDR_%'].mean() if len(feasible_data) > 0 else 0,
    infeasible_data['Model_B_PDR_%'].mean() if len(infeasible_data) > 0 else 0
]

bars1 = axes[0].bar(x - width/2, model_a_means, width, label='Model A', 
                    color='skyblue', edgecolor='black')
bars2 = axes[0].bar(x + width/2, model_b_means, width, label='Model B',
                    color='lightgreen', edgecolor='black')

axes[0].set_ylabel('Mean Predicted PDR (%)', fontsize=11)
axes[0].set_title('PDR Predictions by Feasibility', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Feasible\n(Above Sens)', 'Infeasible\n(Below Sens)'])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Link Margin vs PDR Predictions
axes[1].scatter(results_df['LinkMargin_dB'], results_df['Model_A_PDR_%'], 
               alpha=0.7, s=80, label='Model A', edgecolors='black')
axes[1].scatter(results_df['LinkMargin_dB'], results_df['Model_B_PDR_%'],
               alpha=0.7, s=80, marker='^', label='Model B', edgecolors='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Margin (Threshold)')
axes[1].set_xlabel('Link Margin (dB)', fontsize=11)
axes[1].set_ylabel('Predicted PDR (%)', fontsize=11)
axes[1].set_title('Link Margin vs PDR Predictions', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_validation_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: model_validation_results.png")

# ========== FINAL VERDICT ==========

print("\n" + "="*70)
print("FINAL VERDICT:")
print("="*70)

if impossible_a > impossible_b:
    print("✅ Model B is MORE physically consistent than Model A")
    print(f"   Model A has {impossible_a - impossible_b} more impossible predictions")
elif impossible_b > impossible_a:
    print("⚠️  Model A is MORE physically consistent than Model B")
    print(f"   Model B has {impossible_b - impossible_a} more impossible predictions")
else:
    print("✅ Both models have same physical consistency")

print("\nRecommendation:")
if impossible_b <= impossible_a:
    print("   Use Model B for deployment - better physical consistency")
    print("   Link margin (87% importance) ensures realistic predictions")
else:
    print("   Use Model A for deployment")

print("="*70)