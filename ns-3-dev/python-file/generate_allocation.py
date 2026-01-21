#!/usr/bin/env python3
"""
LoRaWAN Dataset Generation - Allocation Matrix Generator
Goal: Minimize interference by distributing combinations evenly
"""

import numpy as np
import pandas as pd
import random

# Configuration
NUM_NODES = 500
NUM_COMBINATIONS = 90  # 6 DR × 3 Freq × 5 TxPower
NUM_EXPERIMENTS = 90

# Generate all combinations
DR_VALUES = [0, 1, 2, 3, 4, 5]
FREQ_VALUES = [868100000, 868300000, 868500000]  # Hz
TX_POWER_VALUES = [6, 8, 10, 12, 14]  # dBm

combinations = []
for dr in DR_VALUES:
    for freq in FREQ_VALUES:
        for tp in TX_POWER_VALUES:
            combinations.append({
                'combId': len(combinations),
                'dr': dr,
                'freq': freq,
                'txPower': tp
            })

print(f"Total combinations: {len(combinations)}")
assert len(combinations) == NUM_COMBINATIONS

# Save combinations to CSV
df_comb = pd.DataFrame(combinations)
df_comb.to_csv('combinations.csv', index=False)
print("✓ Saved: combinations.csv")

# ========== Generate Allocation Matrix ==========
print(f"\nGenerating allocation matrix...")
print(f"  Nodes: {NUM_NODES}")
print(f"  Experiments: {NUM_EXPERIMENTS}")
print(f"  Reuse per experiment: {NUM_NODES / NUM_COMBINATIONS:.2f}x")

# Matrix: allocation[node_id][experiment_id] = combination_id
allocation = np.zeros((NUM_NODES, NUM_EXPERIMENTS), dtype=int)

# Strategy: For each node, assign all 90 combinations randomly across 90 experiments
for node_id in range(NUM_NODES):
    # Shuffle combinations for this node
    node_combinations = list(range(NUM_COMBINATIONS))
    random.shuffle(node_combinations)
    
    # Assign to experiments
    for exp_id in range(NUM_EXPERIMENTS):
        allocation[node_id][exp_id] = node_combinations[exp_id]

# Verify: Each node uses all 90 combinations exactly once
for node_id in range(NUM_NODES):
    unique_combs = set(allocation[node_id])
    assert len(unique_combs) == NUM_COMBINATIONS, f"Node {node_id} error"

print("✓ Verification passed: Each node uses all 90 combinations")

# Calculate statistics per experiment
print("\n=== Interference Analysis ===")
for exp_id in range(min(5, NUM_EXPERIMENTS)):  # Show first 5
    combs_in_exp = allocation[:, exp_id]
    unique, counts = np.unique(combs_in_exp, return_counts=True)
    
    print(f"Experiment {exp_id+1}:")
    print(f"  Min reuse: {counts.min()}x")
    print(f"  Max reuse: {counts.max()}x")
    print(f"  Avg reuse: {counts.mean():.2f}x")
    print(f"  Std dev: {counts.std():.2f}")

# Save allocation matrix to CSV
print("\nSaving allocation matrix...")
df_allocation = pd.DataFrame(allocation)
df_allocation.columns = [f'exp_{i+1}' for i in range(NUM_EXPERIMENTS)]
df_allocation.index.name = 'node_id'
df_allocation.to_csv('allocation_matrix.csv')
print("✓ Saved: allocation_matrix.csv")

# Create a detailed schedule CSV (easier for C++ to read)
print("\nCreating detailed schedule...")
schedule = []
for exp_id in range(NUM_EXPERIMENTS):
    for node_id in range(NUM_NODES):
        comb_id = allocation[node_id][exp_id]
        comb = combinations[comb_id]
        schedule.append({
            'experiment': exp_id + 1,
            'nodeId': node_id,
            'combId': comb_id,
            'dr': comb['dr'],
            'freq': comb['freq'],
            'txPower': comb['txPower']
        })

df_schedule = pd.DataFrame(schedule)
df_schedule.to_csv('experiment_schedule.csv', index=False)
print("✓ Saved: experiment_schedule.csv")

print("\n=== Generation Complete! ===")
print(f"Total data points: {len(schedule)}")
print(f"\nFiles created:")
print(f"  1. combinations.csv - All 90 combinations")
print(f"  2. allocation_matrix.csv - Matrix view")
print(f"  3. experiment_schedule.csv - Detailed schedule (USE THIS in C++)")