#!/usr/bin/env python3
"""
LoRaWAN Dataset Analysis & Visualization
Custom for your dataset format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 70)
print("         LoRaWAN Dataset Analysis & Visualization")
print("=" * 70)

# ============================================================
# STEP 1: MERGE ALL CSV FILES
# ============================================================

print("\n[1] Merging experiment CSV files...")

# ✅ FIX: Pattern untuk file results_experiment_*.csv
csv_files = []
for i in range(1, 91):
    filename = f'results_experiment_{i}.csv'
    if Path(filename).exists():
        csv_files.append((filename, i))  # Simpan tuple (filename, exp_number)

print(f"Found {len(csv_files)}/{90} CSV files")

if len(csv_files) == 0:
    print("ERROR: No results_experiment_*.csv files found!")
    print("Please check if CSV files are in the current directory")
    exit(1)

# Read and merge all CSVs
dfs = []
for filename, exp_num in csv_files:
    try:
        df = pd.read_csv(filename)
        df['experiment'] = exp_num  # ✅ FIX: Gunakan exp_num langsung
        dfs.append(df)
        if len(dfs) % 10 == 0:
            print(f"  Loaded {len(dfs)}/{len(csv_files)} files...")
    except Exception as e:
        print(f"  Warning: Could not read {filename}: {e}")

# Check if we have any data
if len(dfs) == 0:
    print("ERROR: No valid CSV files could be read!")
    exit(1)

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)
print(f"✓ Merged dataset: {len(df):,} rows")

# Calculate PDR (Packet Delivery Ratio)
df['PDR'] = (df['RxPackets'] / df['TxPackets'] * 100).round(2)
df['PacketsLost'] = df['TxPackets'] - df['RxPackets']

# Save merged dataset
output_file = 'dataset_complete.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# ============================================================
# STEP 2: DATA SUMMARY
# ============================================================

print("\n[2] Dataset Summary...")
print(f"\n{'='*70}")
print(f"Total Records: {len(df):,}")
print(f"Total Experiments: {df['experiment'].nunique()}")
print(f"Total Nodes: {df['node_id'].nunique()}")
print(f"{'='*70}")

print(f"\nData Rate Distribution:")
for dr in sorted(df['dr'].unique()):
    count = len(df[df['dr'] == dr])
    pct = count / len(df) * 100
    print(f"  DR{dr}: {count:>6,} ({pct:>5.2f}%)")

print(f"\nTx Power Distribution:")
for tp in sorted(df['txPower'].unique()):
    count = len(df[df['txPower'] == tp])
    pct = count / len(df) * 100
    print(f"  {tp:>2} dBm: {count:>6,} ({pct:>5.2f}%)")

print(f"\nFrequency Distribution:")
for freq in sorted(df['freq'].unique()):
    count = len(df[df['freq'] == freq])
    pct = count / len(df) * 100
    print(f"  {freq:>5.1f} MHz: {count:>6,} ({pct:>5.2f}%)")

print(f"\nDistance Statistics (meters):")
print(f"  Min:    {df['distance'].min():>8.2f}")
print(f"  Max:    {df['distance'].max():>8.2f}")
print(f"  Mean:   {df['distance'].mean():>8.2f}")
print(f"  Median: {df['distance'].median():>8.2f}")
print(f"  Std:    {df['distance'].std():>8.2f}")

print(f"\nPacket Delivery Ratio (%):")
print(f"  Min:    {df['PDR'].min():>6.2f}")
print(f"  Max:    {df['PDR'].max():>6.2f}")
print(f"  Mean:   {df['PDR'].mean():>6.2f}")
print(f"  Median: {df['PDR'].median():>6.2f}")

print(f"\nTime on Air Statistics (ms):")
print(f"  Avg ToA Min:  {df['AvgToA_ms'].min():>8.2f}")
print(f"  Avg ToA Max:  {df['AvgToA_ms'].max():>8.2f}")
print(f"  Avg ToA Mean: {df['AvgToA_ms'].mean():>8.2f}")

# ============================================================
# STEP 3: VISUALIZATIONS
# ============================================================

print("\n[3] Creating visualizations...")
Path('plots').mkdir(exist_ok=True)

# -------------------- PLOT 1: Parameter Distribution --------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LoRaWAN Parameter Distribution (45,000 Data Points)', 
             fontsize=18, fontweight='bold', y=0.995)

# DR distribution
dr_counts = df['dr'].value_counts().sort_index()
bars1 = axes[0, 0].bar(dr_counts.index, dr_counts.values, 
                       color='steelblue', edgecolor='black', linewidth=1.5)
axes[0, 0].set_xlabel('Data Rate (DR)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Count', fontsize=13, fontweight='bold')
axes[0, 0].set_title('Data Rate Distribution', fontsize=15, fontweight='bold')
axes[0, 0].set_xticks(range(6))
axes[0, 0].set_xticklabels([f'DR{i}' for i in range(6)])
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

# TxPower distribution
tp_counts = df['txPower'].value_counts().sort_index()
bars2 = axes[0, 1].bar(tp_counts.index, tp_counts.values, 
                       color='coral', edgecolor='black', linewidth=1.5)
axes[0, 1].set_xlabel('Tx Power (dBm)', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Count', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Transmission Power Distribution', fontsize=15, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')

# Frequency distribution
freq_counts = df['freq'].value_counts().sort_index()
bars3 = axes[1, 0].bar(range(len(freq_counts)), freq_counts.values,
                       tick_label=[f'{f:.1f}' for f in freq_counts.index],
                       color='mediumseagreen', edgecolor='black', linewidth=1.5)
axes[1, 0].set_xlabel('Frequency (MHz)', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Count', fontsize=13, fontweight='bold')
axes[1, 0].set_title('Frequency Distribution', fontsize=15, fontweight='bold')
for bar in bars3:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

# DR vs TxPower heatmap
combo_counts = df.groupby(['dr', 'txPower']).size().reset_index(name='count')
pivot = combo_counts.pivot(index='dr', columns='txPower', values='count').fillna(0)
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1],
           cbar_kws={'label': 'Count'}, linewidths=0.5)
axes[1, 1].set_xlabel('Tx Power (dBm)', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Data Rate (DR)', fontsize=13, fontweight='bold')
axes[1, 1].set_title('DR vs TxPower Combinations', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/1_parameter_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/1_parameter_distribution.png")
plt.close()

# -------------------- PLOT 2: Distance Analysis --------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distance-based Analysis', fontsize=18, fontweight='bold', y=0.995)

# Distance histogram
axes[0, 0].hist(df['distance'], bins=50, color='skyblue', 
               edgecolor='black', alpha=0.7, linewidth=1.5)
axes[0, 0].axvline(df['distance'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f"Mean: {df['distance'].mean():.1f}m")
axes[0, 0].set_xlabel('Distance (m)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[0, 0].set_title('Node Distance Distribution', fontsize=15, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

# Distance vs DR scatter
colors = plt.cm.viridis(np.linspace(0, 1, 6))
for dr in sorted(df['dr'].unique()):
    dr_data = df[df['dr'] == dr]
    axes[0, 1].scatter(dr_data['distance'], dr_data['PDR'], 
                      alpha=0.4, s=5, label=f'DR{dr}', color=colors[dr])
axes[0, 1].set_xlabel('Distance (m)', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('PDR (%)', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Distance vs PDR (colored by DR)', fontsize=15, fontweight='bold')
axes[0, 1].legend(fontsize=10, ncol=2)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# Distance bins vs PDR
df['distance_bin'] = pd.cut(df['distance'], bins=10)
distance_pdr = df.groupby('distance_bin')['PDR'].mean()
bin_centers = [interval.mid for interval in distance_pdr.index]
axes[1, 0].plot(bin_centers, distance_pdr.values, marker='o', 
               linewidth=2, markersize=8, color='green')
axes[1, 0].set_xlabel('Distance (m)', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Average PDR (%)', fontsize=13, fontweight='bold')
axes[1, 0].set_title('Average PDR vs Distance', fontsize=15, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# Node positions
axes[1, 1].scatter(df['x'], df['y'], c=df['distance'], 
                  cmap='coolwarm', s=1, alpha=0.5)
axes[1, 1].scatter(0, 0, marker='*', s=500, c='red', 
                  edgecolors='black', linewidths=2, label='Gateway', zorder=5)
axes[1, 1].set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
axes[1, 1].set_title('Node Spatial Distribution', fontsize=15, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('plots/2_distance_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/2_distance_analysis.png")
plt.close()

# -------------------- PLOT 3: Performance Metrics --------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Metrics Analysis', fontsize=18, fontweight='bold', y=0.995)

# PDR distribution
axes[0, 0].hist(df['PDR'], bins=50, color='lightgreen', 
               edgecolor='black', alpha=0.7, linewidth=1.5)
axes[0, 0].axvline(df['PDR'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f"Mean: {df['PDR'].mean():.2f}%")
axes[0, 0].set_xlabel('Packet Delivery Ratio (%)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[0, 0].set_title('PDR Distribution', fontsize=15, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

# Packets sent vs received
axes[0, 1].scatter(df['TxPackets'], df['RxPackets'], alpha=0.3, s=10, color='purple')
axes[0, 1].plot([df['TxPackets'].min(), df['TxPackets'].max()],
               [df['TxPackets'].min(), df['TxPackets'].max()],
               'r--', linewidth=2, label='Perfect PDR (100%)')
axes[0, 1].set_xlabel('Transmitted Packets', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Received Packets', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Tx vs Rx Packets', fontsize=15, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# Time on Air distribution
axes[1, 0].hist(df['AvgToA_ms'], bins=50, color='orange', 
               edgecolor='black', alpha=0.7, linewidth=1.5)
axes[1, 0].axvline(df['AvgToA_ms'].mean(), color='blue', linestyle='--',
                  linewidth=2, label=f"Mean: {df['AvgToA_ms'].mean():.2f}ms")
axes[1, 0].set_xlabel('Average Time on Air (ms)', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[1, 0].set_title('Avg ToA Distribution', fontsize=15, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

# Packets lost
lost_counts = df['PacketsLost'].value_counts().sort_index()
axes[1, 1].bar(lost_counts.index, lost_counts.values, 
              color='crimson', edgecolor='black', alpha=0.7, linewidth=1.5)
axes[1, 1].set_xlabel('Packets Lost', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[1, 1].set_title('Packet Loss Distribution', fontsize=15, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/3_performance_metrics.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/3_performance_metrics.png")
plt.close()

# -------------------- PLOT 4: DR Performance Comparison --------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Data Rate Performance Comparison', fontsize=18, fontweight='bold')

# PDR by DR
dr_pdr_data = [df[df['dr'] == dr]['PDR'].values for dr in sorted(df['dr'].unique())]
bp1 = axes[0].boxplot(dr_pdr_data, labels=[f'DR{dr}' for dr in range(6)],
                      patch_artist=True, notch=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(bp1['boxes'])))
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_xlabel('Data Rate', fontsize=13, fontweight='bold')
axes[0].set_ylabel('PDR (%)', fontsize=13, fontweight='bold')
axes[0].set_title('PDR by Data Rate', fontsize=15, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# ToA by DR
dr_toa_data = [df[df['dr'] == dr]['AvgToA_ms'].values for dr in sorted(df['dr'].unique())]
bp2 = axes[1].boxplot(dr_toa_data, labels=[f'DR{dr}' for dr in range(6)],
                      patch_artist=True, notch=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_xlabel('Data Rate', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Avg ToA (ms)', fontsize=13, fontweight='bold')
axes[1].set_title('Time on Air by Data Rate', fontsize=15, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Average PDR per DR
dr_avg_pdr = df.groupby('dr')['PDR'].mean()
bars = axes[2].bar(dr_avg_pdr.index, dr_avg_pdr.values, 
                   color=colors, edgecolor='black', linewidth=1.5)
axes[2].set_xlabel('Data Rate', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Average PDR (%)', fontsize=13, fontweight='bold')
axes[2].set_title('Average PDR per DR', fontsize=15, fontweight='bold')
axes[2].set_xticks(range(6))
axes[2].set_xticklabels([f'DR{i}' for i in range(6)])
for bar in bars:
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/4_dr_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/4_dr_performance.png")
plt.close()

# -------------------- PLOT 5: TxPower Performance --------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Tx Power Performance Comparison', fontsize=18, fontweight='bold')

# PDR by TxPower
tp_pdr_data = [df[df['txPower'] == tp]['PDR'].values for tp in sorted(df['txPower'].unique())]
bp1 = axes[0].boxplot(tp_pdr_data, labels=[f'{tp}dBm' for tp in sorted(df['txPower'].unique())],
                      patch_artist=True, notch=True)
colors = plt.cm.plasma(np.linspace(0, 1, len(bp1['boxes'])))
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_xlabel('Tx Power', fontsize=13, fontweight='bold')
axes[0].set_ylabel('PDR (%)', fontsize=13, fontweight='bold')
axes[0].set_title('PDR by Tx Power', fontsize=15, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Average PDR per TxPower
tp_avg_pdr = df.groupby('txPower')['PDR'].mean()
bars = axes[1].bar(range(len(tp_avg_pdr)), tp_avg_pdr.values,
                   tick_label=[f'{tp}' for tp in tp_avg_pdr.index],
                   color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Tx Power (dBm)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average PDR (%)', fontsize=13, fontweight='bold')
axes[1].set_title('Average PDR per Tx Power', fontsize=15, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# TxPower vs Distance effect
for tp in sorted(df['txPower'].unique()):
    tp_data = df[df['txPower'] == tp]
    tp_data_sorted = tp_data.sort_values('distance')
    # Running average
    window = 500
    tp_data_sorted['PDR_smooth'] = tp_data_sorted['PDR'].rolling(window=window, center=True).mean()
    axes[2].plot(tp_data_sorted['distance'], tp_data_sorted['PDR_smooth'], 
                linewidth=2, label=f'{tp} dBm', alpha=0.8)
axes[2].set_xlabel('Distance (m)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('PDR (%) - Smoothed', fontsize=13, fontweight='bold')
axes[2].set_title('PDR vs Distance by Tx Power', fontsize=15, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/5_txpower_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/5_txpower_performance.png")
plt.close()

# -------------------- PLOT 6: Experiment Progress --------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experiment Progress (90 Experiments)', fontsize=18, fontweight='bold', y=0.995)

# Records per experiment
exp_counts = df.groupby('experiment').size()
axes[0, 0].plot(exp_counts.index, exp_counts.values, marker='o', 
               linewidth=2, markersize=5, color='steelblue')
axes[0, 0].axhline(exp_counts.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {exp_counts.mean():.0f}')
axes[0, 0].set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Number of Records', fontsize=13, fontweight='bold')
axes[0, 0].set_title('Data Points per Experiment', fontsize=15, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')

# PDR evolution
exp_pdr = df.groupby('experiment')['PDR'].mean()
axes[0, 1].plot(exp_pdr.index, exp_pdr.values, marker='o',
               linewidth=2, markersize=5, color='green')
axes[0, 1].axhline(exp_pdr.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Overall Mean: {exp_pdr.mean():.2f}%')
axes[0, 1].set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Average PDR (%)', fontsize=13, fontweight='bold')
axes[0, 1].set_title('PDR Evolution Across Experiments', fontsize=15, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')

# Avg ToA evolution
exp_toa = df.groupby('experiment')['AvgToA_ms'].mean()
axes[1, 0].plot(exp_toa.index, exp_toa.values, marker='o',
               linewidth=2, markersize=5, color='orange')
axes[1, 0].axhline(exp_toa.mean(), color='blue', linestyle='--',
                  linewidth=2, label=f'Overall Mean: {exp_toa.mean():.2f}ms')
axes[1, 0].set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Average ToA (ms)', fontsize=13, fontweight='bold')
axes[1, 0].set_title('ToA Evolution Across Experiments', fontsize=15, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')

# Packets lost per experiment
exp_lost = df.groupby('experiment')['PacketsLost'].sum()
axes[1, 1].bar(exp_lost.index, exp_lost.values, color='crimson', alpha=0.7)
axes[1, 1].axhline(exp_lost.mean(), color='blue', linestyle='--',
                  linewidth=2, label=f'Mean: {exp_lost.mean():.0f}')
axes[1, 1].set_xlabel('Experiment Number', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Total Packets Lost', fontsize=13, fontweight='bold')
axes[1, 1].set_title('Packet Loss per Experiment', fontsize=15, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('plots/6_experiment_progress.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: plots/6_experiment_progress.png")
plt.close()

# ============================================================
# STEP 4: DETAILED STATISTICAL SUMMARY
# ============================================================

print("\n[4] Generating detailed statistical report...")

summary = []
summary.append("=" * 70)
summary.append("           LORAWAN DATASET - STATISTICAL REPORT")
summary.append("=" * 70)
summary.append(f"\nDataset Information:")
summary.append(f"  Total Records:     {len(df):,}")
summary.append(f"  Total Experiments: {df['experiment'].nunique()}")
summary.append(f"  Total Nodes:       {df['node_id'].nunique()}")
summary.append(f"  Records per Node:  {len(df) / df['node_id'].nunique():.1f}")

summary.append("\n" + "-" * 70)
summary.append("PARAMETER DISTRIBUTION")
summary.append("-" * 70)

summary.append("\nData Rate (DR):")
for dr in sorted(df['dr'].unique()):
    count = len(df[df['dr'] == dr])
    pct = count / len(df) * 100
    summary.append(f"  DR{dr}: {count:>7,} ({pct:>5.2f}%)")

summary.append("\nTransmission Power (dBm):")
for tp in sorted(df['txPower'].unique()):
    count = len(df[df['txPower'] == tp])
    pct = count / len(df) * 100
    summary.append(f"  {tp:>2} dBm: {count:>7,} ({pct:>5.2f}%)")

summary.append("\nFrequency (MHz):")
for freq in sorted(df['freq'].unique()):
    count = len(df[df['freq'] == freq])
    pct = count / len(df) * 100
    summary.append(f"  {freq:>5.1f}: {count:>7,} ({pct:>5.2f}%)")

summary.append("\n" + "-" * 70)
summary.append("PERFORMANCE METRICS")
summary.append("-" * 70)

summary.append("\nPacket Delivery Ratio (%):")
summary.append(f"  Min:     {df['PDR'].min():>7.2f}")
summary.append(f"  Max:     {df['PDR'].max():>7.2f}")
summary.append(f"  Mean:    {df['PDR'].mean():>7.2f}")
summary.append(f"  Median:  {df['PDR'].median():>7.2f}")
summary.append(f"  Std Dev: {df['PDR'].std():>7.2f}")

summary.append("\nPacket Statistics:")
summary.append(f"  Total Tx:   {df['TxPackets'].sum():>10,}")
summary.append(f"  Total Rx:   {df['RxPackets'].sum():>10,}")
summary.append(f"  Total Lost: {df['PacketsLost'].sum():>10,}")
summary.append(f"  Overall PDR: {(df['RxPackets'].sum() / df['TxPackets'].sum() * 100):>6.2f}%")

summary.append("\nTime on Air (ms):")
summary.append(f"  Avg ToA Min:  {df['AvgToA_ms'].min():>8.2f}")
summary.append(f"  Avg ToA Max:  {df['AvgToA_ms'].max():>8.2f}")
summary.append(f"  Avg ToA Mean: {df['AvgToA_ms'].mean():>8.2f}")
summary.append(f"  Total ToA:    {df['TotalToA_ms'].sum() / 1000:>8.2f} seconds")

summary.append("\n" + "-" * 70)
summary.append("DISTANCE ANALYSIS")
summary.append("-" * 70)

summary.append("\nDistance Statistics (meters):")
summary.append(f"  Min:     {df['distance'].min():>8.2f}")
summary.append(f"  Max:     {df['distance'].max():>8.2f}")
summary.append(f"  Mean:    {df['distance'].mean():>8.2f}")
summary.append(f"  Median:  {df['distance'].median():>8.2f}")
summary.append(f"  Std Dev: {df['distance'].std():>8.2f}")

summary.append("\n" + "-" * 70)
summary.append("PERFORMANCE BY DATA RATE")
summary.append("-" * 70)

for dr in sorted(df['dr'].unique()):
    dr_data = df[df['dr'] == dr]
    summary.append(f"\nDR{dr}:")
    summary.append(f"  Count:       {len(dr_data):>7,}")
    summary.append(f"  Avg PDR:     {dr_data['PDR'].mean():>7.2f}%")
    summary.append(f"  Avg ToA:     {dr_data['AvgToA_ms'].mean():>7.2f} ms")
    summary.append(f"  Avg Distance:{dr_data['distance'].mean():>7.2f} m")

summary.append("\n" + "-" * 70)
summary.append("PERFORMANCE BY TX POWER")
summary.append("-" * 70)

for tp in sorted(df['txPower'].unique()):
    tp_data = df[df['txPower'] == tp]
    summary.append(f"\n{tp} dBm:")
    summary.append(f"  Count:       {len(tp_data):>7,}")
    summary.append(f"  Avg PDR:     {tp_data['PDR'].mean():>7.2f}%")
    summary.append(f"  Avg Distance:{tp_data['distance'].mean():>7.2f} m")

summary.append("\n" + "=" * 70)
summary.append("END OF REPORT")
summary.append("=" * 70)

# Print and save summary
summary_text = "\n".join(summary)
print(summary_text)

with open('dataset_summary.txt', 'w') as f:
    f.write(summary_text)
print("\n✓ Saved: dataset_summary.txt")

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("                    ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nGenerated Files:")
print(f"  1. dataset_complete.csv      - Merged dataset ({len(df):,} rows)")
print(f"  2. dataset_summary.txt       - Statistical report")
print(f"  3. plots/                    - Visualization images:")
print(f"     - 1_parameter_distribution.png")
print(f"     - 2_distance_analysis.png")
print(f"     - 3_performance_metrics.png")
print(f"     - 4_dr_performance.png")
print(f"     - 5_txpower_performance.png")
print(f"     - 6_experiment_progress.png")
print("\n" + "=" * 70)