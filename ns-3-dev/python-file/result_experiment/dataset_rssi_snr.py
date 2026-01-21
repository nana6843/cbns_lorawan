#!/usr/bin/env python3
"""
Calculate RSSI, SNR, and Sensitivity for LoRaWAN Dataset
Check correlation with PDR (Packet Delivery Ratio)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# ========== CONFIGURATION ==========
PATH_LOSS_EXPONENT = 3.76
REFERENCE_DISTANCE = 1.0
REFERENCE_LOSS = 7.7

BANDWIDTH_HZ = 125000
NOISE_FIGURE_DB = 6
THERMAL_NOISE_DENSITY = -174

NOISE_FLOOR_DBM = THERMAL_NOISE_DENSITY + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB

print("="*70)
print("LoRaWAN RSSI, SNR & Sensitivity Calculator")
print("="*70)
print(f"Path Loss Model:")
print(f"  - Exponent: {PATH_LOSS_EXPONENT}")
print(f"  - Reference: {REFERENCE_LOSS} dB @ {REFERENCE_DISTANCE}m")
print(f"\nReceiver Parameters:")
print(f"  - Bandwidth: {BANDWIDTH_HZ/1000:.0f} kHz")
print(f"  - Noise Figure: {NOISE_FIGURE_DB} dB")
print(f"  - Noise Floor: {NOISE_FLOOR_DBM:.2f} dBm")
print("="*70)

# ========== FUNCTIONS ==========

def calculate_path_loss(distance):
    if distance < REFERENCE_DISTANCE:
        distance = REFERENCE_DISTANCE
    path_loss = REFERENCE_LOSS + 10 * PATH_LOSS_EXPONENT * np.log10(distance / REFERENCE_DISTANCE)
    return path_loss

def calculate_rssi(tx_power_dbm, distance):
    path_loss = calculate_path_loss(distance)
    rssi = tx_power_dbm - path_loss
    return rssi

def calculate_snr(rssi_dbm):
    snr = rssi_dbm - NOISE_FLOOR_DBM
    return snr

def dr_to_sf(dr):
    sf_map = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
    return sf_map.get(dr, 12)

def get_sensitivity(sf):
    sensitivity_map = {
        7: -130.0, 8: -132.5, 9: -135.0,
        10: -137.5, 11: -140.0, 12: -142.5
    }
    return sensitivity_map.get(sf, -130.0)

def get_reception_status(rssi, sensitivity):
    return "ABOVE" if rssi > sensitivity else "BELOW"

# ========== LOAD DATA ==========

input_file = 'dataset_complete.csv'
if len(sys.argv) > 1:
    input_file = sys.argv[1]

print(f"\nLoading data from: {input_file}")

try:
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} records")
except FileNotFoundError:
    print(f"ERROR: File {input_file} not found!")
    sys.exit(1)

# Check required columns
required_cols = ['distance', 'txPower', 'dr']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

# Check if PDR exists
has_pdr = 'RxPackets' in df.columns and 'TxPackets' in df.columns

# ========== CALCULATE RSSI, SNR & SENSITIVITY ==========

print("\nCalculating RSSI, SNR, and Sensitivity...")

df['RSSI'] = df.apply(lambda row: round(calculate_rssi(row['txPower'], row['distance']), 2), axis=1)
df['SNR'] = df.apply(lambda row: round(calculate_snr(row['RSSI']), 2), axis=1)
df['SF'] = df['dr'].apply(dr_to_sf)
df['Sensitivity'] = df['SF'].apply(get_sensitivity)
df['Above_Sensitivity'] = df['RSSI'] > df['Sensitivity']
df['Reception_Status'] = df.apply(lambda row: get_reception_status(row['RSSI'], row['Sensitivity']), axis=1)

# Calculate PDR if available
if has_pdr:
    df['PDR'] = (df['RxPackets'] / df['TxPackets'] * 100).round(2)
    df['PDR'] = df['PDR'].fillna(0)  # Handle division by zero
    print("✓ PDR calculated from RxPackets/TxPackets")

# ========== STATISTICS ==========

print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)

print(f"\nRSSI (dBm):")
print(f"  Min:  {df['RSSI'].min():.2f}")
print(f"  Max:  {df['RSSI'].max():.2f}")
print(f"  Mean: {df['RSSI'].mean():.2f}")
print(f"  Std:  {df['RSSI'].std():.2f}")

print(f"\nSNR (dB):")
print(f"  Min:  {df['SNR'].min():.2f}")
print(f"  Max:  {df['SNR'].max():.2f}")
print(f"  Mean: {df['SNR'].mean():.2f}")
print(f"  Std:  {df['SNR'].std():.2f}")

if has_pdr:
    print(f"\nPDR (%):")
    print(f"  Min:  {df['PDR'].min():.2f}")
    print(f"  Max:  {df['PDR'].max():.2f}")
    print(f"  Mean: {df['PDR'].mean():.2f}")
    print(f"  Std:  {df['PDR'].std():.2f}")

above_count = df['Above_Sensitivity'].sum()
below_count = (~df['Above_Sensitivity']).sum()
print(f"\nReception Status:")
print(f"  ABOVE Sensitivity: {above_count} ({above_count/len(df)*100:.2f}%)")
print(f"  BELOW Sensitivity: {below_count} ({below_count/len(df)*100:.2f}%)")

# ========== CORRELATION WITH PDR ==========

if has_pdr:
    print(f"\n{'='*70}")
    print("CORRELATION WITH PDR")
    print("="*70)
    
    # Overall correlation
    corr_rssi_pdr = df['RSSI'].corr(df['PDR'])
    corr_snr_pdr = df['SNR'].corr(df['PDR'])
    
    print(f"\nPearson Correlation with PDR:")
    print(f"  RSSI vs PDR: {corr_rssi_pdr:.2f}")
    print(f"  SNR vs PDR:  {corr_snr_pdr:.2f}")
    
    # Compare ABOVE vs BELOW sensitivity with PDR
    df_above = df[df['Above_Sensitivity']]
    df_below = df[~df['Above_Sensitivity']]
    
    print(f"\nPDR Statistics by Reception Status:")
    print(f"  ABOVE Sensitivity: PDR = {df_above['PDR'].mean():.2f}% (n={len(df_above)})")
    print(f"  BELOW Sensitivity: PDR = {df_below['PDR'].mean():.2f}% (n={len(df_below)})")
    
    # Check match: if ABOVE sensitivity but PDR = 0
    mismatch_above = df[(df['Above_Sensitivity']) & (df['PDR'] == 0)]
    # Check match: if BELOW sensitivity but PDR > 0
    mismatch_below = df[(~df['Above_Sensitivity']) & (df['PDR'] > 0)]
    
    print(f"\nMismatch Analysis:")
    print(f"  ABOVE but PDR=0:    {len(mismatch_above)} ({len(mismatch_above)/len(df)*100:.2f}%)")
    print(f"  BELOW but PDR>0:    {len(mismatch_below)} ({len(mismatch_below)/len(df)*100:.2f}%)")
    
    # Expected match: ABOVE with PDR>0, BELOW with PDR=0
    match_correct = df[((df['Above_Sensitivity']) & (df['PDR'] > 0)) | 
                      ((~df['Above_Sensitivity']) & (df['PDR'] == 0))]
    print(f"  Expected Match:     {len(match_correct)} ({len(match_correct)/len(df)*100:.2f}%)")

# ========== PER SF ANALYSIS ==========

print(f"\n{'='*70}")
print("STATISTICS PER SPREADING FACTOR")
print("="*70)

for sf in sorted(df['SF'].unique()):
    df_sf = df[df['SF'] == sf]
    sensitivity = get_sensitivity(sf)
    above = df_sf['Above_Sensitivity'].sum()
    below = (~df_sf['Above_Sensitivity']).sum()
    
    print(f"\nSF{sf} (DR{12-sf}) | Sensitivity: {sensitivity:.2f} dBm")
    print(f"  Records: {len(df_sf)}")
    print(f"  RSSI:  {df_sf['RSSI'].mean():.2f} ± {df_sf['RSSI'].std():.2f} dBm")
    print(f"  SNR:   {df_sf['SNR'].mean():.2f} ± {df_sf['SNR'].std():.2f} dB")
    
    if has_pdr:
        print(f"  PDR:   {df_sf['PDR'].mean():.2f} ± {df_sf['PDR'].std():.2f} %")
    
    print(f"  ABOVE: {above} ({above/len(df_sf)*100:.2f}%)")
    print(f"  BELOW: {below} ({below/len(df_sf)*100:.2f}%)")

# ========== CRITICAL CASES ==========

df_below_sens = df[~df['Above_Sensitivity']].sort_values('RSSI')

if len(df_below_sens) > 0:
    print(f"\n{'='*70}")
    print(f"CRITICAL: {len(df_below_sens)} records BELOW sensitivity")
    print("="*70)
    
    if has_pdr:
        print(f"\nTop 10 worst cases (with PDR):")
        cols = ['node_id', 'distance', 'SF', 'txPower', 'RSSI', 'Sensitivity', 'PDR']
    else:
        print(f"\nTop 10 worst cases:")
        cols = ['node_id', 'distance', 'SF', 'txPower', 'RSSI', 'Sensitivity']
    
    available_cols = [col for col in cols if col in df_below_sens.columns]
    print(df_below_sens[available_cols].head(10).to_string(index=False))

# ========== SAVE RESULTS ==========

output_file = input_file.replace('.csv', '_with_rssi_snr.csv')

# Reorder columns: put RSSI, SNR, Sensitivity after dr/freq columns
cols = df.columns.tolist()
new_cols = [col for col in cols if col not in ['RSSI', 'SNR', 'SF', 'Sensitivity', 'Above_Sensitivity', 'Reception_Status', 'PDR']]

# Insert new columns after freq (if exists) or dr
insert_pos = new_cols.index('freq') + 1 if 'freq' in new_cols else (new_cols.index('dr') + 1 if 'dr' in new_cols else len(new_cols))
new_cols.insert(insert_pos, 'RSSI')
new_cols.insert(insert_pos + 1, 'SNR')
new_cols.insert(insert_pos + 2, 'SF')
new_cols.insert(insert_pos + 3, 'Sensitivity')
new_cols.insert(insert_pos + 4, 'Above_Sensitivity')
new_cols.insert(insert_pos + 5, 'Reception_Status')

if has_pdr and 'PDR' not in new_cols:
    new_cols.insert(insert_pos + 6, 'PDR')

df = df[new_cols]
df.to_csv(output_file, index=False)
print(f"\n✓ Saved results to: {output_file}")

# Summary per SF
summary_data = []
for sf in sorted(df['SF'].unique()):
    df_sf = df[df['SF'] == sf]
    row = {
        'SF': sf,
        'DR': 12 - sf,
        'Sensitivity_dBm': f"{get_sensitivity(sf):.2f}",
        'Records': len(df_sf),
        'RSSI_mean': f"{df_sf['RSSI'].mean():.2f}",
        'RSSI_std': f"{df_sf['RSSI'].std():.2f}",
        'SNR_mean': f"{df_sf['SNR'].mean():.2f}",
        'SNR_std': f"{df_sf['SNR'].std():.2f}",
        'Above_Count': df_sf['Above_Sensitivity'].sum(),
        'Above_Percent': f"{df_sf['Above_Sensitivity'].sum()/len(df_sf)*100:.2f}"
    }
    
    if has_pdr:
        row['PDR_mean'] = f"{df_sf['PDR'].mean():.2f}"
        row['PDR_std'] = f"{df_sf['PDR'].std():.2f}"
    
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
summary_file = input_file.replace('.csv', '_summary.csv')
df_summary.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")

# ========== VISUALIZATION ==========

if has_pdr:
    print("\nGenerating visualizations with PDR analysis...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: RSSI vs Distance (colored by Above/Below)
    ax1 = plt.subplot(2, 3, 1)
    colors = ['green' if above else 'red' for above in df['Above_Sensitivity']]
    ax1.scatter(df['distance'], df['RSSI'], c=colors, s=15, alpha=0.5, edgecolors='none')
    for sf in sorted(df['SF'].unique()):
        ax1.axhline(get_sensitivity(sf), linestyle=':', linewidth=0.8, alpha=0.4)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('RSSI (dBm)')
    ax1.set_title('RSSI vs Distance\n(Green=ABOVE, Red=BELOW)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SNR vs Distance
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(df['distance'], df['SNR'], c=df['SF'], cmap='viridis', s=15, alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='SF')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('SNR vs Distance')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PDR vs RSSI
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(df['RSSI'], df['PDR'], c=colors, s=15, alpha=0.5)
    ax3.set_xlabel('RSSI (dBm)')
    ax3.set_ylabel('PDR (%)')
    ax3.set_title(f'PDR vs RSSI\n(Correlation: {corr_rssi_pdr:.2f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Above/Below Status
    ax4 = plt.subplot(2, 3, 4)
    status_counts = df['Reception_Status'].value_counts()
    colors4 = ['green', 'red']
    bars = ax4.bar(status_counts.index, status_counts.values, color=colors4, edgecolor='black', alpha=0.8)
    ax4.set_ylabel('Count')
    ax4.set_title('Reception Status')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 5: PDR comparison
    ax5 = plt.subplot(2, 3, 5)
    pdr_above = df[df['Above_Sensitivity']]['PDR'].mean()
    pdr_below = df[~df['Above_Sensitivity']]['PDR'].mean()
    bars = ax5.bar(['ABOVE\nSensitivity', 'BELOW\nSensitivity'], [pdr_above, pdr_below], 
                   color=['green', 'red'], edgecolor='black', alpha=0.8)
    ax5.set_ylabel('Average PDR (%)')
    ax5.set_title('PDR: ABOVE vs BELOW Sensitivity')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [pdr_above, pdr_below]):
        ax5.text(bar.get_x() + bar.get_width()/2., val, f'{val:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 6: PDR per SF
    ax6 = plt.subplot(2, 3, 6)
    sf_values = sorted(df['SF'].unique())
    pdr_per_sf = [df[df['SF'] == sf]['PDR'].mean() for sf in sf_values]
    ax6.bar([f'SF{sf}' for sf in sf_values], pdr_per_sf, color='steelblue', edgecolor='black', alpha=0.8)
    ax6.set_ylabel('Average PDR (%)')
    ax6.set_title('PDR per Spreading Factor')
    ax6.grid(True, alpha=0.3, axis='y')
    for i, (sf, pdr) in enumerate(zip(sf_values, pdr_per_sf)):
        ax6.text(i, pdr, f'{pdr:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    viz_file = 'rssi_snr_pdr_analysis.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_file}")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)
print(f"Files created:")
print(f"  1. {output_file}")
print(f"  2. {summary_file}")
if has_pdr:
    print(f"  3. {viz_file}")
print("="*70)