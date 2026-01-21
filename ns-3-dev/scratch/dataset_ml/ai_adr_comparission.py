import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

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

def create_comparison_plots(ai_data, conv_data):
    """Create comprehensive comparison visualizations"""
    
    # Combine data for easier plotting
    combined_data = pd.concat([ai_data, conv_data], ignore_index=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. PDR Comparison - Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(ai_data))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ai_data['avg_pdr'], width, 
                    label='AI-Powered ADR', color='#2ecc71', alpha=0.8,
                    yerr=ai_data['std_pdr'], capsize=5)
    bars2 = ax1.bar(x + width/2, conv_data['avg_pdr'], width,
                    label='Conventional ADR', color='#e74c3c', alpha=0.8,
                    yerr=conv_data['std_pdr'], capsize=5)
    
    ax1.set_xlabel('Number of Nodes', fontweight='bold')
    ax1.set_ylabel('Average PDR (%)', fontweight='bold')
    ax1.set_title('Packet Delivery Ratio Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ai_data['num_nodes'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Energy Consumption Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars3 = ax2.bar(x - width/2, ai_data['avg_energy_consumed'], width,
                    label='AI-Powered ADR', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, conv_data['avg_energy_consumed'], width,
                    label='Conventional ADR', color='#f39c12', alpha=0.8)
    
    ax2.set_xlabel('Number of Nodes', fontweight='bold')
    ax2.set_ylabel('Avg Energy Consumed (J)', fontweight='bold')
    ax2.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ai_data['num_nodes'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 3. Network PDR Trend
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(ai_data['num_nodes'], ai_data['network_pdr'], 
             marker='o', linewidth=2.5, markersize=8, 
             label='AI-Powered ADR', color='#2ecc71')
    ax3.plot(conv_data['num_nodes'], conv_data['network_pdr'],
             marker='s', linewidth=2.5, markersize=8,
             label='Conventional ADR', color='#e74c3c')
    
    ax3.set_xlabel('Number of Nodes', fontweight='bold')
    ax3.set_ylabel('Network PDR (%)', fontweight='bold')
    ax3.set_title('Network-wide PDR Scalability', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # 4. Total Energy Consumed
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(ai_data['num_nodes'], ai_data['total_energy_consumed']/1000,
             marker='o', linewidth=2.5, markersize=8,
             label='AI-Powered ADR', color='#3498db')
    ax4.plot(conv_data['num_nodes'], conv_data['total_energy_consumed']/1000,
             marker='s', linewidth=2.5, markersize=8,
             label='Conventional ADR', color='#f39c12')
    
    ax4.set_xlabel('Number of Nodes', fontweight='bold')
    ax4.set_ylabel('Total Energy Consumed (kJ)', fontweight='bold')
    ax4.set_title('Total Network Energy Consumption', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. PDR Improvement Percentage
    ax5 = plt.subplot(2, 3, 5)
    pdr_improvement = ((ai_data['avg_pdr'].values - conv_data['avg_pdr'].values) / 
                       conv_data['avg_pdr'].values * 100)
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in pdr_improvement]
    bars5 = ax5.bar(x, pdr_improvement, color=colors, alpha=0.8)
    
    ax5.set_xlabel('Number of Nodes', fontweight='bold')
    ax5.set_ylabel('PDR Improvement (%)', fontweight='bold')
    ax5.set_title('AI-Powered ADR vs Conventional ADR\nPDR Improvement', 
                  fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(ai_data['num_nodes'])
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 6. Energy Efficiency (PDR per Joule)
    ax6 = plt.subplot(2, 3, 6)
    ai_efficiency = ai_data['avg_pdr'] / ai_data['avg_energy_consumed']
    conv_efficiency = conv_data['avg_pdr'] / conv_data['avg_energy_consumed']
    
    bars6 = ax6.bar(x - width/2, ai_efficiency, width,
                    label='AI-Powered ADR', color='#9b59b6', alpha=0.8)
    bars7 = ax6.bar(x + width/2, conv_efficiency, width,
                    label='Conventional ADR', color='#16a085', alpha=0.8)
    
    ax6.set_xlabel('Number of Nodes', fontweight='bold')
    ax6.set_ylabel('Energy Efficiency (PDR% / J)', fontweight='bold')
    ax6.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(ai_data['num_nodes'])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/adr_comparison_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    print("Saved: adr_comparison_comprehensive.png")
    plt.close()

def create_detailed_metrics_plot(ai_data, conv_data):
    """Create additional detailed comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Box plot comparison for PDR distribution
    ax1 = axes[0, 0]
    positions = []
    data_to_plot = []
    labels = []
    colors = []
    
    for i, num_nodes in enumerate(sorted(ai_data['num_nodes'].unique())):
        positions.extend([i*3, i*3+1])
        data_to_plot.extend([
            [ai_data[ai_data['num_nodes']==num_nodes]['avg_pdr'].values[0]],
            [conv_data[conv_data['num_nodes']==num_nodes]['avg_pdr'].values[0]]
        ])
        labels.extend([f'AI-{num_nodes}', f'Conv-{num_nodes}'])
        colors.extend(['#2ecc71', '#e74c3c'])
    
    # Simpler plot
    x_pos = np.arange(len(ai_data))
    ax1.scatter(x_pos - 0.15, ai_data['avg_pdr'], s=100, 
                color='#2ecc71', alpha=0.6, label='AI-Powered ADR', marker='o')
    ax1.scatter(x_pos + 0.15, conv_data['avg_pdr'], s=100,
                color='#e74c3c', alpha=0.6, label='Conventional ADR', marker='s')
    
    ax1.set_xlabel('Network Size Configuration', fontweight='bold')
    ax1.set_ylabel('PDR (%)', fontweight='bold')
    ax1.set_title('PDR Distribution Across Network Sizes', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{n} nodes' for n in ai_data['num_nodes']])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Remaining Energy Comparison
    ax2 = axes[0, 1]
    width = 0.35
    x_pos = np.arange(len(ai_data))
    
    ax2.bar(x_pos - width/2, ai_data['avg_remaining_energy'], width,
            label='AI-Powered ADR', color='#27ae60', alpha=0.8)
    ax2.bar(x_pos + width/2, conv_data['avg_remaining_energy'], width,
            label='Conventional ADR', color='#c0392b', alpha=0.8)
    
    ax2.set_xlabel('Number of Nodes', fontweight='bold')
    ax2.set_ylabel('Avg Remaining Energy (J)', fontweight='bold')
    ax2.set_title('Average Remaining Energy per Node', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ai_data['num_nodes'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Success Rate (Total Packets)
    ax3 = axes[1, 0]
    
    ax3.plot(ai_data['num_nodes'], ai_data['total_rx_packets']/1000,
             marker='o', linewidth=2.5, markersize=8,
             label='AI-Powered ADR (Received)', color='#2ecc71')
    ax3.plot(conv_data['num_nodes'], conv_data['total_rx_packets']/1000,
             marker='s', linewidth=2.5, markersize=8,
             label='Conventional ADR (Received)', color='#e74c3c')
    
    ax3.set_xlabel('Number of Nodes', fontweight='bold')
    ax3.set_ylabel('Total Received Packets (×1000)', fontweight='bold')
    ax3.set_title('Total Successfully Received Packets', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Energy Savings
    ax4 = axes[1, 1]
    energy_savings = ((conv_data['avg_energy_consumed'].values - 
                      ai_data['avg_energy_consumed'].values) / 
                     conv_data['avg_energy_consumed'].values * 100)
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in energy_savings]
    bars = ax4.bar(x_pos, energy_savings, color=colors, alpha=0.8)
    
    ax4.set_xlabel('Number of Nodes', fontweight='bold')
    ax4.set_ylabel('Energy Savings (%)', fontweight='bold')
    ax4.set_title('Energy Savings with AI-Powered ADR', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(ai_data['num_nodes'])
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/adr_detailed_metrics.png',
                dpi=300, bbox_inches='tight')
    print("Saved: adr_detailed_metrics.png")
    plt.close()

def create_summary_table(ai_data, conv_data):
    """Create a summary comparison table"""
    
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
    table.set_fontsize(10)
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
    
    plt.savefig('/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/adr_summary_table.png',
                dpi=300, bbox_inches='tight')
    print("Saved: adr_summary_table.png")
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
    
    print("Loading and processing data...")
    print("-" * 80)
    
    # Load data
    ai_data = load_and_process_data(AI_FILES, 'AI-Powered ADR')
    conv_data = load_and_process_data(CONVENTIONAL_FILES, 'Conventional ADR')
    
    if ai_data.empty or conv_data.empty:
        print("Error: Could not load data. Please check file paths.")
        return
    
    print(f"✓ Loaded {len(ai_data)} AI-Powered ADR datasets")
    print(f"✓ Loaded {len(conv_data)} Conventional ADR datasets")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    create_comparison_plots(ai_data, conv_data)
    create_detailed_metrics_plot(ai_data, conv_data)
    create_summary_table(ai_data, conv_data)
    
    # Print statistics
    print_statistics(ai_data, conv_data)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. adr_comparison_comprehensive.png - Main comparison charts")
    print("  2. adr_detailed_metrics.png - Detailed analysis plots")
    print("  3. adr_summary_table.png - Performance summary table")
    print("\n")

if __name__ == "__main__":
    main()