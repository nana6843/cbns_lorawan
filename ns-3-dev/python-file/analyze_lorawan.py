#!/usr/bin/env python3
"""
LoRaWAN RSSI and SNR Analysis from Simulation Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path loss model parameters (same as simulation)
PATH_LOSS_EXPONENT = 3.76
PATH_LOSS_AT_1M = 7.7
NOISE_FLOOR = -123.0  # dBm for 125 kHz bandwidth

def calculate_rssi(tx_power_dbm, distance_m):
    """Calculate RSSI using log-distance path loss model"""
    if distance_m < 1.0:
        distance_m = 1.0
    
    path_loss = PATH_LOSS_AT_1M + 10.0 * PATH_LOSS_EXPONENT * np.log10(distance_m)
    rssi = tx_power_dbm - path_loss
    return rssi

def calculate_snr(rssi_dbm):
    """Calculate SNR from RSSI"""
    snr = rssi_dbm - NOISE_FLOOR
    return snr

def analyze_network():
    """Main analysis function"""
    
    print("=" * 60)
    print("LoRaWAN Network Analysis - RSSI and SNR Calculator")
    print("=" * 60)
    
    # Read node information
    try:
        nodes = pd.read_csv('node_info.csv')
    except FileNotFoundError:
        print("ERROR: node_info.csv not found!")
        print("Make sure you ran the simulation first.")
        return
    
    print(f"\nTotal Nodes: {len(nodes)}\n")
    
    # Calculate RSSI and SNR for each node
    nodes['RSSI'] = nodes.apply(
        lambda row: calculate_rssi(row['TxPower'], row['Distance']), 
        axis=1
    )
    nodes['SNR'] = nodes['RSSI'].apply(calculate_snr)
    
    # Display results
    print("=" * 100)
    print(f"{'Node':<6} {'Distance':<12} {'TxPower':<10} {'RSSI':<12} {'SNR':<12} {'Status':<15}")
    print("=" * 100)
    
    for _, row in nodes.iterrows():
        # Determine status based on SNR
        if row['SNR'] > 20:
            status = "Excellent"
        elif row['SNR'] > 10:
            status = "Good"
        elif row['SNR'] > 0:
            status = "Fair"
        else:
            status = "Poor"
        
        print(f"{row['NodeId']:<6} {row['Distance']:<12.2f} {row['TxPower']:<10.2f} "
              f"{row['RSSI']:<12.2f} {row['SNR']:<12.2f} {status:<15}")
    
    print("=" * 100)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Network Statistics")
    print("=" * 60)
    print(f"Average RSSI:     {nodes['RSSI'].mean():.2f} dBm")
    print(f"Min RSSI:         {nodes['RSSI'].min():.2f} dBm")
    print(f"Max RSSI:         {nodes['RSSI'].max():.2f} dBm")
    print(f"Std Dev RSSI:     {nodes['RSSI'].std():.2f} dB")
    print()
    print(f"Average SNR:      {nodes['SNR'].mean():.2f} dB")
    print(f"Min SNR:          {nodes['SNR'].min():.2f} dB")
    print(f"Max SNR:          {nodes['SNR'].max():.2f} dB")
    print(f"Std Dev SNR:      {nodes['SNR'].std():.2f} dB")
    print()
    print(f"Average Distance: {nodes['Distance'].mean():.2f} m")
    print(f"Max Distance:     {nodes['Distance'].max():.2f} m")
    
    # Signal quality distribution
    excellent = len(nodes[nodes['SNR'] > 20])
    good = len(nodes[(nodes['SNR'] >= 10) & (nodes['SNR'] <= 20)])
    fair = len(nodes[(nodes['SNR'] >= 0) & (nodes['SNR'] < 10)])
    poor = len(nodes[nodes['SNR'] < 0])
    
    print("\n" + "=" * 60)
    print("Signal Quality Distribution")
    print("=" * 60)
    print(f"Excellent (SNR > 20 dB):   {excellent} nodes ({excellent/len(nodes)*100:.1f}%)")
    print(f"Good (10-20 dB):           {good} nodes ({good/len(nodes)*100:.1f}%)")
    print(f"Fair (0-10 dB):            {fair} nodes ({fair/len(nodes)*100:.1f}%)")
    print(f"Poor (< 0 dB):             {poor} nodes ({poor/len(nodes)*100:.1f}%)")
    
    # Save results
    nodes.to_csv('rssi_snr_results.csv', index=False)
    print(f"\nResults saved to: rssi_snr_results.csv")
    
    # Generate plots
    generate_plots(nodes)
    
    return nodes

def generate_plots(nodes):
    """Generate visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LoRaWAN Network Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: RSSI vs Distance
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(nodes['Distance'], nodes['RSSI'], 
                          c=nodes['SNR'], cmap='RdYlGn', 
                          s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('RSSI (dBm)', fontsize=12)
    ax1.set_title('RSSI vs Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    # Add node labels
    for _, row in nodes.iterrows():
        ax1.annotate(f"N{row['NodeId']}", 
                    (row['Distance'], row['RSSI']),
                    fontsize=8, ha='right')
    
    # Plot 2: SNR vs Distance
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(nodes['Distance'], nodes['SNR'], 
                          c=nodes['SNR'], cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='SNR = 0 dB')
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=1, label='SNR = 10 dB')
    ax2.axhline(y=20, color='green', linestyle='--', linewidth=1, label='SNR = 20 dB')
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('SNR (dB)', fontsize=12)
    ax2.set_title('SNR vs Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: RSSI Distribution
    ax3 = axes[1, 0]
    ax3.hist(nodes['RSSI'], bins=15, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax3.axvline(nodes['RSSI'].mean(), color='red', 
               linestyle='--', linewidth=2, 
               label=f'Mean: {nodes["RSSI"].mean():.2f} dBm')
    ax3.set_xlabel('RSSI (dBm)', fontsize=12)
    ax3.set_ylabel('Number of Nodes', fontsize=12)
    ax3.set_title('RSSI Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: SNR Distribution
    ax4 = axes[1, 1]
    ax4.hist(nodes['SNR'], bins=15, color='green', 
            edgecolor='black', alpha=0.7)
    ax4.axvline(nodes['SNR'].mean(), color='red', 
               linestyle='--', linewidth=2, 
               label=f'Mean: {nodes["SNR"].mean():.2f} dB')
    ax4.axvline(0, color='orange', linestyle='--', 
               linewidth=2, label='SNR = 0 dB')
    ax4.set_xlabel('SNR (dB)', fontsize=12)
    ax4.set_ylabel('Number of Nodes', fontsize=12)
    ax4.set_title('SNR Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('lorawan_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to: lorawan_analysis.png")
    plt.show()

if __name__ == "__main__":
    nodes = analyze_network()