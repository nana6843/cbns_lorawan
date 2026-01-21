import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Output directory
OUTPUT_DIR = '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/'

# File paths for Random Forest (previously AI-Powered ADR)
RF_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_200_5000_1.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_400_5000_1.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_600_5000_1.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_800_5000_1.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_1000_5000_1.csv',
}

# File paths for LightGBM (NEW)
LGBM_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_200_5000_1.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_400_5000_1.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_600_5000_1.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_800_5000_1.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_lgbm_1000_5000_1.csv',
}

# File paths for Conventional ADR
CONVENTIONAL_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_200_5000_1_adr.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_400_5000_1_adr.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_600_5000_1_adr.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_800_5000_1_adr.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_1000_5000_1_adr.csv',
}

def load_and_process_data(file_paths, method_name):
    """Load all CSV files and calculate metrics"""
    results = []
    
    for num_nodes, filepath in file_paths.items():
        try:
            df = pd.read_csv(filepath)
            
            # Calculate PDR (Packet Delivery Ratio)
            df['PDR'] = (df['RxPackets'] / df['TxPackets']) * 100
            
            # Calculate Energy Consumption (Initial - Remaining)
            # Assuming initial energy is 20000 J
            df['EnergyConsumed_J'] = 20000 - df['RemainingEnergy_J']
            
            # Calculate metrics
            metrics = {
                'num_nodes': num_nodes,
                'method': method_name,
                'avg_pdr': df['PDR'].mean(),
                'std_pdr': df['PDR'].std(),
                'median_pdr': df['PDR'].median(),
                'min_pdr': df['PDR'].min(),
                'max_pdr': df['PDR'].max(),
                'avg_energy_consumed': df['EnergyConsumed_J'].mean(),
                'std_energy_consumed': df['EnergyConsumed_J'].std(),
                'total_energy_consumed': df['EnergyConsumed_J'].sum(),
                'avg_remaining_energy': df['RemainingEnergy_J'].mean(),
                'total_tx_packets': df['TxPackets'].sum(),
                'total_rx_packets': df['RxPackets'].sum(),
                'network_pdr': (df['RxPackets'].sum() / df['TxPackets'].sum()) * 100,
                'avg_toa': df['AvgToA_ms'].mean(),
            }
            
            results.append(metrics)
            
        except FileNotFoundError:
            print(f"Warning: File not found - {filepath}")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    return pd.DataFrame(results)


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS (3-WAY COMPARISON)
# ============================================================================

def plot1_pdr_comparison_bar(rf_data, lgbm_data, conv_data):
    """1. PDR Comparison - Bar Chart (3 methods)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rf_data))
    width = 0.25
    
    bars1 = ax.bar(x - width, conv_data['avg_pdr'], width, 
                   label='Conventional ADR', color='#e74c3c', alpha=0.8,
                   yerr=conv_data['std_pdr'], capsize=4)
    bars2 = ax.bar(x, rf_data['avg_pdr'], width,
                   label='Random Forest', color='#2ecc71', alpha=0.8,
                   yerr=rf_data['std_pdr'], capsize=4)
    bars3 = ax.bar(x + width, lgbm_data['avg_pdr'], width,
                   label='LightGBM', color='#3498db', alpha=0.8,
                   yerr=lgbm_data['std_pdr'], capsize=4)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('Packet Delivery Ratio Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '1_pdr_comparison_bar_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot2_energy_consumption_bar(rf_data, lgbm_data, conv_data):
    """2. Energy Consumption Comparison (3 methods)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rf_data))
    width = 0.25
    
    bars1 = ax.bar(x - width, conv_data['avg_energy_consumed'], width,
                   label='Conventional ADR', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, rf_data['avg_energy_consumed'], width,
                   label='Random Forest', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, lgbm_data['avg_energy_consumed'], width,
                   label='LightGBM', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Energy Consumed (J)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '2_energy_consumption_bar_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot3_network_pdr_trend(rf_data, lgbm_data, conv_data):
    """3. Network PDR Trend (3 methods)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(conv_data['num_nodes'], conv_data['network_pdr'],
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR', color='#e74c3c')
    ax.plot(rf_data['num_nodes'], rf_data['network_pdr'], 
            marker='o', linewidth=2.5, markersize=10, 
            label='Random Forest', color='#2ecc71')
    ax.plot(lgbm_data['num_nodes'], lgbm_data['network_pdr'],
            marker='^', linewidth=2.5, markersize=10,
            label='LightGBM', color='#3498db')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Network PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('Network-wide PDR Scalability', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '3_network_pdr_trend_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot4_total_energy_consumed(rf_data, lgbm_data, conv_data):
    """4. Total Energy Consumed (3 methods)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(conv_data['num_nodes'], conv_data['total_energy_consumed']/1000,
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR', color='#e74c3c')
    ax.plot(rf_data['num_nodes'], rf_data['total_energy_consumed']/1000,
            marker='o', linewidth=2.5, markersize=10,
            label='Random Forest', color='#2ecc71')
    ax.plot(lgbm_data['num_nodes'], lgbm_data['total_energy_consumed']/1000,
            marker='^', linewidth=2.5, markersize=10,
            label='LightGBM', color='#3498db')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Energy Consumed (kJ)', fontweight='bold', fontsize=12)
    ax.set_title('Total Network Energy Consumption', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '4_total_energy_consumed_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot5_pdr_improvement(rf_data, lgbm_data, conv_data):
    """5. PDR Improvement vs Conventional (RF and LightGBM)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rf_data))
    width = 0.35
    
    rf_improvement = ((rf_data['avg_pdr'].values - conv_data['avg_pdr'].values) / 
                      conv_data['avg_pdr'].values * 100)
    lgbm_improvement = ((lgbm_data['avg_pdr'].values - conv_data['avg_pdr'].values) / 
                        conv_data['avg_pdr'].values * 100)
    
    bars1 = ax.bar(x - width/2, rf_improvement, width,
                   label='Random Forest', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, lgbm_improvement, width,
                   label='LightGBM', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('PDR Improvement vs Conventional (%)', fontweight='bold', fontsize=12)
    ax.set_title('ML Methods PDR Improvement over Conventional ADR', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '5_pdr_improvement_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot6_energy_efficiency(rf_data, lgbm_data, conv_data):
    """6. Energy Efficiency Comparison (PDR per Joule)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rf_data))
    width = 0.25
    
    conv_efficiency = conv_data['avg_pdr'] / conv_data['avg_energy_consumed']
    rf_efficiency = rf_data['avg_pdr'] / rf_data['avg_energy_consumed']
    lgbm_efficiency = lgbm_data['avg_pdr'] / lgbm_data['avg_energy_consumed']
    
    bars1 = ax.bar(x - width, conv_efficiency, width,
                   label='Conventional ADR', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, rf_efficiency, width,
                   label='Random Forest', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, lgbm_efficiency, width,
                   label='LightGBM', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Energy Efficiency (PDR% / J)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '6_energy_efficiency_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot7_pdr_distribution(rf_data, lgbm_data, conv_data):
    """7. PDR Distribution with Error Bars"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(rf_data))
    offset = 0.2
    
    ax.scatter(x_pos - offset, conv_data['avg_pdr'], s=150, 
               color='#e74c3c', alpha=0.7, label='Conventional ADR', marker='s',
               edgecolors='black', linewidth=1.5)
    ax.scatter(x_pos, rf_data['avg_pdr'], s=150,
               color='#2ecc71', alpha=0.7, label='Random Forest', marker='o',
               edgecolors='black', linewidth=1.5)
    ax.scatter(x_pos + offset, lgbm_data['avg_pdr'], s=150,
               color='#3498db', alpha=0.7, label='LightGBM', marker='^',
               edgecolors='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_pos - offset, conv_data['avg_pdr'], yerr=conv_data['std_pdr'],
                fmt='none', ecolor='#c0392b', capsize=5, alpha=0.5)
    ax.errorbar(x_pos, rf_data['avg_pdr'], yerr=rf_data['std_pdr'],
                fmt='none', ecolor='#27ae60', capsize=5, alpha=0.5)
    ax.errorbar(x_pos + offset, lgbm_data['avg_pdr'], yerr=lgbm_data['std_pdr'],
                fmt='none', ecolor='#2980b9', capsize=5, alpha=0.5)
    
    ax.set_xlabel('Network Size Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('PDR Distribution Across Network Sizes', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n} nodes' for n in rf_data['num_nodes']])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '7_pdr_distribution_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot8_remaining_energy(rf_data, lgbm_data, conv_data):
    """8. Remaining Energy Comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(rf_data))
    width = 0.25
    
    bars1 = ax.bar(x_pos - width, conv_data['avg_remaining_energy'], width,
                   label='Conventional ADR', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x_pos, rf_data['avg_remaining_energy'], width,
                   label='Random Forest', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x_pos + width, lgbm_data['avg_remaining_energy'], width,
                   label='LightGBM', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Remaining Energy (J)', fontweight='bold', fontsize=12)
    ax.set_title('Average Remaining Energy per Node', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '8_remaining_energy_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot9_total_received_packets(rf_data, lgbm_data, conv_data):
    """9. Total Successfully Received Packets"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(conv_data['num_nodes'], conv_data['total_rx_packets']/1000,
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR', color='#e74c3c')
    ax.plot(rf_data['num_nodes'], rf_data['total_rx_packets']/1000,
            marker='o', linewidth=2.5, markersize=10,
            label='Random Forest', color='#2ecc71')
    ax.plot(lgbm_data['num_nodes'], lgbm_data['total_rx_packets']/1000,
            marker='^', linewidth=2.5, markersize=10,
            label='LightGBM', color='#3498db')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Received Packets (×1000)', fontweight='bold', fontsize=12)
    ax.set_title('Total Successfully Received Packets', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '9_total_received_packets_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot10_energy_savings(rf_data, lgbm_data, conv_data):
    """10. Energy Savings vs Conventional (RF and LightGBM)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(rf_data))
    width = 0.35
    
    rf_savings = ((conv_data['avg_energy_consumed'].values - 
                   rf_data['avg_energy_consumed'].values) / 
                  conv_data['avg_energy_consumed'].values * 100)
    lgbm_savings = ((conv_data['avg_energy_consumed'].values - 
                     lgbm_data['avg_energy_consumed'].values) / 
                    conv_data['avg_energy_consumed'].values * 100)
    
    bars1 = ax.bar(x_pos - width/2, rf_savings, width,
                   label='Random Forest', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, lgbm_savings, width,
                   label='LightGBM', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Energy Savings vs Conventional (%)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Savings with ML Methods', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rf_data['num_nodes'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '10_energy_savings_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot11_ml_comparison(rf_data, lgbm_data):
    """11. Direct Comparison: Random Forest vs LightGBM"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(rf_data))
    width = 0.35
    
    # PDR Comparison
    ax1.bar(x - width/2, rf_data['avg_pdr'], width,
            label='Random Forest', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, lgbm_data['avg_pdr'], width,
            label='LightGBM', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Average PDR (%)', fontweight='bold', fontsize=12)
    ax1.set_title('PDR: RF vs LightGBM', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rf_data['num_nodes'])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Energy Comparison
    ax2.bar(x - width/2, rf_data['avg_energy_consumed'], width,
            label='Random Forest', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width/2, lgbm_data['avg_energy_consumed'], width,
            label='LightGBM', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Avg Energy Consumed (J)', fontweight='bold', fontsize=12)
    ax2.set_title('Energy: RF vs LightGBM', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rf_data['num_nodes'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Machine Learning Methods Direct Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = OUTPUT_DIR + '11_ml_direct_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot12_summary_table(rf_data, lgbm_data, conv_data):
    """12. Summary Comparison Table (3 methods)"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Metric', '200 Nodes', '400 Nodes', '600 Nodes', 
                       '800 Nodes', '1000 Nodes'])
    
    # PDR rows
    table_data.append(['Conventional PDR (%)',
                      *[f"{val:.2f}" for val in conv_data['avg_pdr'].values]])
    table_data.append(['Random Forest PDR (%)', 
                      *[f"{val:.2f}" for val in rf_data['avg_pdr'].values]])
    table_data.append(['LightGBM PDR (%)',
                      *[f"{val:.2f}" for val in lgbm_data['avg_pdr'].values]])
    table_data.append(['RF Improvement vs Conv (%)',
                      *[f"{((r-c)/c*100):+.2f}" for r, c in 
                        zip(rf_data['avg_pdr'].values, conv_data['avg_pdr'].values)]])
    table_data.append(['LGBM Improvement vs Conv (%)',
                      *[f"{((l-c)/c*100):+.2f}" for l, c in 
                        zip(lgbm_data['avg_pdr'].values, conv_data['avg_pdr'].values)]])
    
    # Energy rows
    table_data.append(['', '', '', '', '', ''])  # Separator
    table_data.append(['Conventional Energy (J)',
                      *[f"{val:.2f}" for val in conv_data['avg_energy_consumed'].values]])
    table_data.append(['Random Forest Energy (J)',
                      *[f"{val:.2f}" for val in rf_data['avg_energy_consumed'].values]])
    table_data.append(['LightGBM Energy (J)',
                      *[f"{val:.2f}" for val in lgbm_data['avg_energy_consumed'].values]])
    table_data.append(['RF Energy Savings (%)',
                      *[f"{((c-r)/c*100):+.2f}" for r, c in 
                        zip(rf_data['avg_energy_consumed'].values, 
                            conv_data['avg_energy_consumed'].values)]])
    table_data.append(['LGBM Energy Savings (%)',
                      *[f"{((c-l)/c*100):+.2f}" for l, c in 
                        zip(lgbm_data['avg_energy_consumed'].values, 
                            conv_data['avg_energy_consumed'].values)]])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # First column
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            elif 'Conventional' in table_data[i][0]:
                cell.set_facecolor('#fadbd8')
            elif 'Random Forest' in table_data[i][0]:
                cell.set_facecolor('#d5f4e6')
            elif 'LightGBM' in table_data[i][0]:
                cell.set_facecolor('#d6eaf8')
            elif 'Improvement' in table_data[i][0] or 'Savings' in table_data[i][0]:
                cell.set_facecolor('#fff9e6')
    
    plt.title('Three-Way Performance Comparison: Conventional vs RF vs LightGBM',
              fontsize=16, fontweight='bold', pad=20)
    
    output_file = OUTPUT_DIR + '12_summary_table_3methods.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_statistics(rf_data, lgbm_data, conv_data):
    """Print detailed statistics for 3 methods"""
    
    print("\n" + "="*90)
    print("THREE-WAY COMPARISON: Conventional ADR vs Random Forest vs LightGBM")
    print("="*90)
    
    for idx, num_nodes in enumerate(rf_data['num_nodes'].values):
        print(f"\n{'─'*90}")
        print(f"Network Size: {num_nodes} Nodes")
        print(f"{'─'*90}")
        
        conv_row = conv_data.iloc[idx]
        rf_row = rf_data.iloc[idx]
        lgbm_row = lgbm_data.iloc[idx]
        
        print(f"\n📊 PACKET DELIVERY RATIO (PDR)")
        print(f"  Conventional ADR:  {conv_row['avg_pdr']:.2f}% (±{conv_row['std_pdr']:.2f})")
        print(f"  Random Forest:     {rf_row['avg_pdr']:.2f}% (±{rf_row['std_pdr']:.2f})")
        print(f"  LightGBM:          {lgbm_row['avg_pdr']:.2f}% (±{lgbm_row['std_pdr']:.2f})")
        
        rf_improvement = ((rf_row['avg_pdr'] - conv_row['avg_pdr']) / 
                         conv_row['avg_pdr'] * 100)
        lgbm_improvement = ((lgbm_row['avg_pdr'] - conv_row['avg_pdr']) / 
                           conv_row['avg_pdr'] * 100)
        print(f"  → RF vs Conv:      {rf_improvement:+.2f}%")
        print(f"  → LGBM vs Conv:    {lgbm_improvement:+.2f}%")
        
        print(f"\n⚡ ENERGY CONSUMPTION")
        print(f"  Conventional ADR:  {conv_row['avg_energy_consumed']:.2f} J")
        print(f"  Random Forest:     {rf_row['avg_energy_consumed']:.2f} J")
        print(f"  LightGBM:          {lgbm_row['avg_energy_consumed']:.2f} J")
        
        rf_savings = ((conv_row['avg_energy_consumed'] - 
                      rf_row['avg_energy_consumed']) / 
                     conv_row['avg_energy_consumed'] * 100)
        lgbm_savings = ((conv_row['avg_energy_consumed'] - 
                        lgbm_row['avg_energy_consumed']) / 
                       conv_row['avg_energy_consumed'] * 100)
        print(f"  → RF Savings:      {rf_savings:+.2f}%")
        print(f"  → LGBM Savings:    {lgbm_savings:+.2f}%")
        
        print(f"\n📡 NETWORK STATISTICS")
        print(f"  Conv Network PDR:  {conv_row['network_pdr']:.2f}%")
        print(f"  RF Network PDR:    {rf_row['network_pdr']:.2f}%")
        print(f"  LGBM Network PDR:  {lgbm_row['network_pdr']:.2f}%")
    
    # Overall summary
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY (Average Across All Network Sizes)")
    print(f"{'='*90}")
    
    rf_avg_improvement = ((rf_data['avg_pdr'].mean() - conv_data['avg_pdr'].mean()) /
                          conv_data['avg_pdr'].mean() * 100)
    lgbm_avg_improvement = ((lgbm_data['avg_pdr'].mean() - conv_data['avg_pdr'].mean()) /
                            conv_data['avg_pdr'].mean() * 100)
    
    rf_avg_savings = ((conv_data['avg_energy_consumed'].mean() - 
                      rf_data['avg_energy_consumed'].mean()) /
                     conv_data['avg_energy_consumed'].mean() * 100)
    lgbm_avg_savings = ((conv_data['avg_energy_consumed'].mean() - 
                        lgbm_data['avg_energy_consumed'].mean()) /
                       conv_data['avg_energy_consumed'].mean() * 100)
    
    print(f"\n✅ Random Forest PDR Improvement:    {rf_avg_improvement:+.2f}%")
    print(f"✅ LightGBM PDR Improvement:         {lgbm_avg_improvement:+.2f}%")
    print(f"✅ Random Forest Energy Savings:     {rf_avg_savings:+.2f}%")
    print(f"✅ LightGBM Energy Savings:          {lgbm_avg_savings:+.2f}%")
    print(f"\n✅ Conventional Average PDR:         {conv_data['avg_pdr'].mean():.2f}%")
    print(f"✅ Random Forest Average PDR:        {rf_data['avg_pdr'].mean():.2f}%")
    print(f"✅ LightGBM Average PDR:             {lgbm_data['avg_pdr'].mean():.2f}%")
    
    # Winner determination
    print(f"\n{'='*90}")
    print("BEST PERFORMER")
    print(f"{'='*90}")
    
    best_pdr = max(rf_data['avg_pdr'].mean(), lgbm_data['avg_pdr'].mean())
    best_energy = min(rf_data['avg_energy_consumed'].mean(), 
                     lgbm_data['avg_energy_consumed'].mean())
    
    pdr_winner = "Random Forest" if rf_data['avg_pdr'].mean() > lgbm_data['avg_pdr'].mean() else "LightGBM"
    energy_winner = "Random Forest" if rf_data['avg_energy_consumed'].mean() < lgbm_data['avg_energy_consumed'].mean() else "LightGBM"
    
    print(f"\n🏆 Best PDR Performance:        {pdr_winner} ({best_pdr:.2f}%)")
    print(f"🏆 Best Energy Performance:     {energy_winner} ({best_energy:.2f} J)")
    print()


def main():
    """Main execution function"""
    
    print("="*90)
    print("THREE-WAY ADR COMPARISON - INDIVIDUAL PLOTS GENERATOR")
    print("Conventional ADR vs Random Forest vs LightGBM")
    print("="*90)
    print("\nLoading and processing data...")
    print("-" * 90)
    
    # Load data
    rf_data = load_and_process_data(RF_FILES, 'Random Forest')
    lgbm_data = load_and_process_data(LGBM_FILES, 'LightGBM')
    conv_data = load_and_process_data(CONVENTIONAL_FILES, 'Conventional ADR')
    
    if rf_data.empty or lgbm_data.empty or conv_data.empty:
        print("Error: Could not load all datasets. Please check file paths.")
        if rf_data.empty:
            print("  ✗ Random Forest data missing")
        if lgbm_data.empty:
            print("  ✗ LightGBM data missing")
        if conv_data.empty:
            print("  ✗ Conventional ADR data missing")
        return
    
    print(f"✓ Loaded {len(rf_data)} Random Forest datasets")
    print(f"✓ Loaded {len(lgbm_data)} LightGBM datasets")
    print(f"✓ Loaded {len(conv_data)} Conventional ADR datasets")
    
    # Generate individual visualizations
    print("\nGenerating individual plots...")
    print("-" * 90)
    
    plot1_pdr_comparison_bar(rf_data, lgbm_data, conv_data)
    plot2_energy_consumption_bar(rf_data, lgbm_data, conv_data)
    plot3_network_pdr_trend(rf_data, lgbm_data, conv_data)
    plot4_total_energy_consumed(rf_data, lgbm_data, conv_data)
    plot5_pdr_improvement(rf_data, lgbm_data, conv_data)
    plot6_energy_efficiency(rf_data, lgbm_data, conv_data)
    plot7_pdr_distribution(rf_data, lgbm_data, conv_data)
    plot8_remaining_energy(rf_data, lgbm_data, conv_data)
    plot9_total_received_packets(rf_data, lgbm_data, conv_data)
    plot10_energy_savings(rf_data, lgbm_data, conv_data)
    plot11_ml_comparison(rf_data, lgbm_data)
    plot12_summary_table(rf_data, lgbm_data, conv_data)
    
    # Print statistics
    print_statistics(rf_data, lgbm_data, conv_data)
    
    print("\n" + "="*90)
    print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*90)
    print("\nGenerated 12 individual plot files:")
    print("  1. 1_pdr_comparison_bar_3methods.png        - PDR comparison (3 bars)")
    print("  2. 2_energy_consumption_bar_3methods.png    - Energy comparison (3 bars)")
    print("  3. 3_network_pdr_trend_3methods.png         - PDR scalability (3 lines)")
    print("  4. 4_total_energy_consumed_3methods.png     - Total energy (3 lines)")
    print("  5. 5_pdr_improvement_3methods.png           - ML improvements vs Conventional")
    print("  6. 6_energy_efficiency_3methods.png         - Efficiency comparison")
    print("  7. 7_pdr_distribution_3methods.png          - PDR distribution with errors")
    print("  8. 8_remaining_energy_3methods.png          - Remaining energy")
    print("  9. 9_total_received_packets_3methods.png    - Total RX packets")
    print(" 10. 10_energy_savings_3methods.png           - Energy savings vs Conventional")
    print(" 11. 11_ml_direct_comparison.png              - RF vs LightGBM head-to-head")
    print(" 12. 12_summary_table_3methods.png            - Complete summary table")
    print("\n")


if __name__ == "__main__":
    main()