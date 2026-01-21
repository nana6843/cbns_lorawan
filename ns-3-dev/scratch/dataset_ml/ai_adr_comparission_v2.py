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

# File paths
AI_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_200_5000_1.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_400_5000_1.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_600_5000_1.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_800_5000_1.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/results_experiment_1000_5000_1.csv',
}

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
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================================

def plot1_pdr_comparison_bar(ai_data, conv_data):
    """1. PDR Comparison - Bar Chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ai_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ai_data['avg_pdr'], width, 
                   label='AI-Powered ADR', color='#2ecc71', alpha=0.8,
                   yerr=ai_data['std_pdr'], capsize=5)
    bars2 = ax.bar(x + width/2, conv_data['avg_pdr'], width,
                   label='Conventional ADR', color='#e74c3c', alpha=0.8,
                   yerr=conv_data['std_pdr'], capsize=5)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('Packet Delivery Ratio Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '1_pdr_comparison_bar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot2_energy_consumption_bar(ai_data, conv_data):
    """2. Energy Consumption Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ai_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ai_data['avg_energy_consumed'], width,
                   label='AI-Powered ADR', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, conv_data['avg_energy_consumed'], width,
                   label='Conventional ADR', color='#f39c12', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Energy Consumed (J)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '2_energy_consumption_bar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot3_network_pdr_trend(ai_data, conv_data):
    """3. Network PDR Trend"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ai_data['num_nodes'], ai_data['network_pdr'], 
            marker='o', linewidth=2.5, markersize=10, 
            label='AI-Powered ADR', color='#2ecc71')
    ax.plot(conv_data['num_nodes'], conv_data['network_pdr'],
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR', color='#e74c3c')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Network PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('Network-wide PDR Scalability', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '3_network_pdr_trend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot4_total_energy_consumed(ai_data, conv_data):
    """4. Total Energy Consumed"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ai_data['num_nodes'], ai_data['total_energy_consumed']/1000,
            marker='o', linewidth=2.5, markersize=10,
            label='AI-Powered ADR', color='#3498db')
    ax.plot(conv_data['num_nodes'], conv_data['total_energy_consumed']/1000,
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR', color='#f39c12')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Energy Consumed (kJ)', fontweight='bold', fontsize=12)
    ax.set_title('Total Network Energy Consumption', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '4_total_energy_consumed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot5_pdr_improvement(ai_data, conv_data):
    """5. PDR Improvement Percentage"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ai_data))
    pdr_improvement = ((ai_data['avg_pdr'].values - conv_data['avg_pdr'].values) / 
                       conv_data['avg_pdr'].values * 100)
    
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in pdr_improvement]
    bars = ax.bar(x, pdr_improvement, color=colors, alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('PDR Improvement (%)', fontweight='bold', fontsize=12)
    ax.set_title('AI-Powered ADR vs Conventional ADR - PDR Improvement', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '5_pdr_improvement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot6_energy_efficiency(ai_data, conv_data):
    """6. Energy Efficiency (PDR per Joule)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(ai_data))
    width = 0.35
    
    ai_efficiency = ai_data['avg_pdr'] / ai_data['avg_energy_consumed']
    conv_efficiency = conv_data['avg_pdr'] / conv_data['avg_energy_consumed']
    
    bars1 = ax.bar(x - width/2, ai_efficiency, width,
                   label='AI-Powered ADR', color='#9b59b6', alpha=0.8)
    bars2 = ax.bar(x + width/2, conv_efficiency, width,
                   label='Conventional ADR', color='#16a085', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Energy Efficiency (PDR% / J)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '6_energy_efficiency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot7_pdr_distribution(ai_data, conv_data):
    """7. PDR Distribution Across Network Sizes"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(ai_data))
    
    ax.scatter(x_pos - 0.15, ai_data['avg_pdr'], s=150, 
               color='#2ecc71', alpha=0.7, label='AI-Powered ADR', marker='o',
               edgecolors='black', linewidth=1.5)
    ax.scatter(x_pos + 0.15, conv_data['avg_pdr'], s=150,
               color='#e74c3c', alpha=0.7, label='Conventional ADR', marker='s',
               edgecolors='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_pos - 0.15, ai_data['avg_pdr'], yerr=ai_data['std_pdr'],
                fmt='none', ecolor='#27ae60', capsize=5, alpha=0.5)
    ax.errorbar(x_pos + 0.15, conv_data['avg_pdr'], yerr=conv_data['std_pdr'],
                fmt='none', ecolor='#c0392b', capsize=5, alpha=0.5)
    
    ax.set_xlabel('Network Size Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('PDR (%)', fontweight='bold', fontsize=12)
    ax.set_title('PDR Distribution Across Network Sizes', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{n} nodes' for n in ai_data['num_nodes']])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '7_pdr_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot8_remaining_energy(ai_data, conv_data):
    """8. Remaining Energy Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(ai_data))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, ai_data['avg_remaining_energy'], width,
                   label='AI-Powered ADR', color='#27ae60', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, conv_data['avg_remaining_energy'], width,
                   label='Conventional ADR', color='#c0392b', alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg Remaining Energy (J)', fontweight='bold', fontsize=12)
    ax.set_title('Average Remaining Energy per Node', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '8_remaining_energy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot9_total_received_packets(ai_data, conv_data):
    """9. Total Successfully Received Packets"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ai_data['num_nodes'], ai_data['total_rx_packets']/1000,
            marker='o', linewidth=2.5, markersize=10,
            label='AI-Powered ADR (Received)', color='#2ecc71')
    ax.plot(conv_data['num_nodes'], conv_data['total_rx_packets']/1000,
            marker='s', linewidth=2.5, markersize=10,
            label='Conventional ADR (Received)', color='#e74c3c')
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Received Packets (×1000)', fontweight='bold', fontsize=12)
    ax.set_title('Total Successfully Received Packets', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '9_total_received_packets.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot10_energy_savings(ai_data, conv_data):
    """10. Energy Savings with AI-Powered ADR"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(ai_data))
    energy_savings = ((conv_data['avg_energy_consumed'].values - 
                      ai_data['avg_energy_consumed'].values) / 
                     conv_data['avg_energy_consumed'].values * 100)
    
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in energy_savings]
    bars = ax.bar(x_pos, energy_savings, color=colors, alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    ax.set_ylabel('Energy Savings (%)', fontweight='bold', fontsize=12)
    ax.set_title('Energy Savings with AI-Powered ADR', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ai_data['num_nodes'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR + '10_energy_savings.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot11_summary_table(ai_data, conv_data):
    """11. Summary Comparison Table"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Metric', '200 Nodes', '400 Nodes', '600 Nodes', 
                       '800 Nodes', '1000 Nodes'])
    
    # PDR rows
    table_data.append(['AI-Powered PDR (%)', 
                      *[f"{val:.2f}" for val in ai_data['avg_pdr'].values]])
    table_data.append(['Conventional PDR (%)',
                      *[f"{val:.2f}" for val in conv_data['avg_pdr'].values]])
    table_data.append(['PDR Improvement (%)',
                      *[f"{((a-c)/c*100):+.2f}" for a, c in 
                        zip(ai_data['avg_pdr'].values, conv_data['avg_pdr'].values)]])
    
    # Energy rows
    table_data.append(['', '', '', '', '', ''])  # Separator
    table_data.append(['AI Energy Consumed (J)',
                      *[f"{val:.2f}" for val in ai_data['avg_energy_consumed'].values]])
    table_data.append(['Conv Energy Consumed (J)',
                      *[f"{val:.2f}" for val in conv_data['avg_energy_consumed'].values]])
    table_data.append(['Energy Savings (%)',
                      *[f"{((c-a)/c*100):+.2f}" for a, c in 
                        zip(ai_data['avg_energy_consumed'].values, 
                            conv_data['avg_energy_consumed'].values)]])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
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
            elif 'AI-Powered' in table_data[i][0]:
                cell.set_facecolor('#d5f4e6')
            elif 'Conventional' in table_data[i][0]:
                cell.set_facecolor('#fadbd8')
            elif 'Improvement' in table_data[i][0] or 'Savings' in table_data[i][0]:
                cell.set_facecolor('#fff9e6')
    
    plt.title('AI-Powered ADR vs Conventional ADR - Performance Summary',
              fontsize=16, fontweight='bold', pad=20)
    
    output_file = OUTPUT_DIR + '11_summary_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_statistics(ai_data, conv_data):
    """Print detailed statistics"""
    
    print("\n" + "="*80)
    print("AI-POWERED ADR vs CONVENTIONAL ADR - DETAILED COMPARISON")
    print("="*80)
    
    for idx, num_nodes in enumerate(ai_data['num_nodes'].values):
        print(f"\n{'─'*80}")
        print(f"Network Size: {num_nodes} Nodes")
        print(f"{'─'*80}")
        
        ai_row = ai_data.iloc[idx]
        conv_row = conv_data.iloc[idx]
        
        print(f"\n📊 PACKET DELIVERY RATIO (PDR)")
        print(f"  AI-Powered ADR:    {ai_row['avg_pdr']:.2f}% (±{ai_row['std_pdr']:.2f})")
        print(f"  Conventional ADR:  {conv_row['avg_pdr']:.2f}% (±{conv_row['std_pdr']:.2f})")
        pdr_improvement = ((ai_row['avg_pdr'] - conv_row['avg_pdr']) / 
                          conv_row['avg_pdr'] * 100)
        print(f"  → Improvement:     {pdr_improvement:+.2f}%")
        
        print(f"\n⚡ ENERGY CONSUMPTION")
        print(f"  AI-Powered ADR:    {ai_row['avg_energy_consumed']:.2f} J")
        print(f"  Conventional ADR:  {conv_row['avg_energy_consumed']:.2f} J")
        energy_savings = ((conv_row['avg_energy_consumed'] - 
                          ai_row['avg_energy_consumed']) / 
                         conv_row['avg_energy_consumed'] * 100)
        print(f"  → Energy Savings:  {energy_savings:+.2f}%")
        
        print(f"\n📡 NETWORK STATISTICS")
        print(f"  AI Network PDR:    {ai_row['network_pdr']:.2f}%")
        print(f"  Conv Network PDR:  {conv_row['network_pdr']:.2f}%")
        print(f"  AI Total RX:       {ai_row['total_rx_packets']:,} packets")
        print(f"  Conv Total RX:     {conv_row['total_rx_packets']:,} packets")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    avg_pdr_improvement = ((ai_data['avg_pdr'].mean() - conv_data['avg_pdr'].mean()) /
                           conv_data['avg_pdr'].mean() * 100)
    avg_energy_savings = ((conv_data['avg_energy_consumed'].mean() - 
                          ai_data['avg_energy_consumed'].mean()) /
                         conv_data['avg_energy_consumed'].mean() * 100)
    
    print(f"\n✅ Average PDR Improvement:    {avg_pdr_improvement:+.2f}%")
    print(f"✅ Average Energy Savings:     {avg_energy_savings:+.2f}%")
    print(f"✅ AI Average PDR:             {ai_data['avg_pdr'].mean():.2f}%")
    print(f"✅ Conventional Average PDR:   {conv_data['avg_pdr'].mean():.2f}%")
    print()


def main():
    """Main execution function"""
    
    print("="*80)
    print("ADR COMPARISON - INDIVIDUAL PLOTS GENERATOR")
    print("="*80)
    print("\nLoading and processing data...")
    print("-" * 80)
    
    # Load data
    ai_data = load_and_process_data(AI_FILES, 'AI-Powered ADR')
    conv_data = load_and_process_data(CONVENTIONAL_FILES, 'Conventional ADR')
    
    if ai_data.empty or conv_data.empty:
        print("Error: Could not load data. Please check file paths.")
        return
    
    print(f"✓ Loaded {len(ai_data)} AI-Powered ADR datasets")
    print(f"✓ Loaded {len(conv_data)} Conventional ADR datasets")
    
    # Generate individual visualizations
    print("\nGenerating individual plots...")
    print("-" * 80)
    
    plot1_pdr_comparison_bar(ai_data, conv_data)
    plot2_energy_consumption_bar(ai_data, conv_data)
    plot3_network_pdr_trend(ai_data, conv_data)
    plot4_total_energy_consumed(ai_data, conv_data)
    plot5_pdr_improvement(ai_data, conv_data)
    plot6_energy_efficiency(ai_data, conv_data)
    plot7_pdr_distribution(ai_data, conv_data)
    plot8_remaining_energy(ai_data, conv_data)
    plot9_total_received_packets(ai_data, conv_data)
    plot10_energy_savings(ai_data, conv_data)
    plot11_summary_table(ai_data, conv_data)
    
    # Print statistics
    print_statistics(ai_data, conv_data)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated 11 individual plot files:")
    print("  1. 1_pdr_comparison_bar.png")
    print("  2. 2_energy_consumption_bar.png")
    print("  3. 3_network_pdr_trend.png")
    print("  4. 4_total_energy_consumed.png")
    print("  5. 5_pdr_improvement.png")
    print("  6. 6_energy_efficiency.png")
    print("  7. 7_pdr_distribution.png")
    print("  8. 8_remaining_energy.png")
    print("  9. 9_total_received_packets.png")
    print(" 10. 10_energy_savings.png")
    print(" 11. 11_summary_table.png")
    print("\n")


if __name__ == "__main__":
    main()