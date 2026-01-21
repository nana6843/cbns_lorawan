#!/usr/bin/env python3
"""
RF-MORA: Parameter Allocation Comparison
AI-Powered Schedule vs Conventional ADR Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CONFIGURATION - EDIT YOUR FILE PATHS HERE
# ============================================================================

# AI-Powered Schedule files (has: experiment, nodeId, combId, dr, freq, txPower)
AI_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_200_5000.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_400_5000.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_600_5000.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_800_5000.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset_ml/experiment_schedule_1000_5000.csv',
}

# Conventional ADR Results files (has: node_id, dr, freq, txPower, TxPackets, RxPackets, etc.)
CONVENTIONAL_FILES = {
    200: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_200_5000_1_adr.csv',
    400: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_400_5000_1_adr.csv',
    600: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_600_5000_1_adr.csv',
    800: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_800_5000_1_adr.csv',
    1000: '/home/nru/CBNS_NEW2/ns-3-dev/scratch/adr-baseline/results_experiment_1000_5000_1_adr.csv',
}

# Output directory
OUTPUT_DIR = './'

# ============================================================================
# END CONFIGURATION
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*100)
print("RF-MORA: PARAMETER ALLOCATION COMPARISON")
print("AI-Powered Schedule vs Conventional ADR Results")
print("="*100)
print()


class ParameterComparison:
    def __init__(self, ai_files, conv_files, output_dir='./'):
        self.ai_files = ai_files
        self.conv_files = conv_files
        self.output_dir = output_dir
        self.device_counts = sorted(ai_files.keys())
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loading data for {len(self.device_counts)} scenarios...\n")
        
        self.ai_data = {}
        self.conv_data = {}
        self.combined_data = {}
        
        for n_devices in self.device_counts:
            print(f"{n_devices} devices:")
            print(f"  AI Schedule: {os.path.basename(ai_files[n_devices])}")
            print(f"  Conventional: {os.path.basename(conv_files[n_devices])}")
            
            try:
                # Load AI schedule
                ai_df = pd.read_csv(ai_files[n_devices])
                # Rename nodeId to node_id for consistency
                if 'nodeId' in ai_df.columns:
                    ai_df = ai_df.rename(columns={'nodeId': 'node_id'})
                ai_df['ADR_Type'] = 'AI-Powered'
                ai_df['nDevices'] = n_devices
                
                # Load Conventional results
                conv_df = pd.read_csv(conv_files[n_devices])
                conv_df['ADR_Type'] = 'Conventional'
                conv_df['nDevices'] = n_devices
                
                self.ai_data[n_devices] = ai_df
                self.conv_data[n_devices] = conv_df
                
                # Combine for visualization
                # Keep only common columns
                common_cols = ['node_id', 'dr', 'freq', 'txPower', 'ADR_Type', 'nDevices']
                ai_subset = ai_df[common_cols].copy()
                conv_subset = conv_df[common_cols].copy()
                self.combined_data[n_devices] = pd.concat([ai_subset, conv_subset], ignore_index=True)
                
                print(f"  ✓ Loaded: {len(ai_df)} AI nodes, {len(conv_df)} conventional nodes")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                raise
        
        print(f"\n✓ Successfully loaded all data!")
    
    def calculate_metrics(self):
        """Calculate parameter allocation metrics"""
        print("\nCalculating parameter allocation metrics...")
        all_metrics = {}
        
        for n_devices in self.device_counts:
            all_metrics[n_devices] = {}
            
            for name, data in [('AI-Powered', self.ai_data[n_devices]), 
                              ('Conventional', self.conv_data[n_devices])]:
                
                metrics = {
                    'nDevices': n_devices,
                    'Total_Nodes': len(data),
                    
                    # Data Rate Statistics
                    'Avg_DR': data['dr'].mean(),
                    'DR_Std': data['dr'].std(),
                    'DR_Min': data['dr'].min(),
                    'DR_Max': data['dr'].max(),
                    'DR_Distribution': data['dr'].value_counts().to_dict(),
                    
                    # TX Power Statistics
                    'Avg_TxPower': data['txPower'].mean(),
                    'TxPower_Std': data['txPower'].std(),
                    'TxPower_Min': data['txPower'].min(),
                    'TxPower_Max': data['txPower'].max(),
                    'TxPower_Distribution': data['txPower'].value_counts().to_dict(),
                    
                    # Frequency Statistics
                    'Freq_Distribution': data['freq'].value_counts().to_dict(),
                    'Unique_Freqs': data['freq'].nunique(),
                    
                    # Resource Efficiency Indicators
                    'High_DR_Nodes': (data['dr'] >= 4).sum(),  # DR 4-5 are higher rates
                    'Low_Power_Nodes': (data['txPower'] <= 10).sum(),  # Lower power
                    'Optimal_Config': ((data['dr'] >= 3) & (data['txPower'] <= 12)).sum(),
                }
                
                all_metrics[n_devices][name] = metrics
        
        print("✓ Metrics calculated!")
        return all_metrics
    
    def print_summary_table(self, metrics):
        """Print summary comparison table"""
        print("\n" + "="*120)
        print("PARAMETER ALLOCATION COMPARISON - ALL SCENARIOS")
        print("="*120)
        
        print(f"\n{'nDevices':<12} {'Metric':<35} {'AI-Powered':<25} {'Conventional':<25} {'Difference':<15}")
        print("-"*120)
        
        for n_devices in self.device_counts:
            ai = metrics[n_devices]['AI-Powered']
            conv = metrics[n_devices]['Conventional']
            
            print(f"\n{'='*120}")
            print(f"{n_devices} DEVICES".center(120))
            print(f"{'='*120}")
            
            # Data Rate
            dr_diff = ai['Avg_DR'] - conv['Avg_DR']
            print(f"{'':<12} {'Average Data Rate':<35} {ai['Avg_DR']:<25.2f} {conv['Avg_DR']:<25.2f} {dr_diff:>+14.2f}")
            print(f"{'':<12} {'DR Range':<35} {ai['DR_Min']}-{ai['DR_Max']:<22} {conv['DR_Min']}-{conv['DR_Max']:<22}")
            print(f"{'':<12} {'High DR Nodes (DR≥4)':<35} {ai['High_DR_Nodes']:<25} {conv['High_DR_Nodes']:<25} {ai['High_DR_Nodes']-conv['High_DR_Nodes']:>+14}")
            
            # TX Power
            power_diff = ai['Avg_TxPower'] - conv['Avg_TxPower']
            print(f"{'':<12} {'Average TX Power (dBm)':<35} {ai['Avg_TxPower']:<25.2f} {conv['Avg_TxPower']:<25.2f} {power_diff:>+14.2f}")
            print(f"{'':<12} {'Power Range (dBm)':<35} {ai['TxPower_Min']}-{ai['TxPower_Max']:<22} {conv['TxPower_Min']}-{conv['TxPower_Max']:<22}")
            print(f"{'':<12} {'Low Power Nodes (≤10dBm)':<35} {ai['Low_Power_Nodes']:<25} {conv['Low_Power_Nodes']:<25} {ai['Low_Power_Nodes']-conv['Low_Power_Nodes']:>+14}")
            
            # Frequency
            print(f"{'':<12} {'Unique Frequencies':<35} {ai['Unique_Freqs']:<25} {conv['Unique_Freqs']:<25}")
            
            # Efficiency
            print(f"{'':<12} {'Optimal Config Nodes':<35} {ai['Optimal_Config']:<25} {conv['Optimal_Config']:<25} {ai['Optimal_Config']-conv['Optimal_Config']:>+14}")
        
        print("\n" + "="*120)
    
    def create_visualizations(self, metrics):
        """Create all visualizations"""
        print("\n" + "="*100)
        print("GENERATING VISUALIZATIONS")
        print("="*100)
        
        print("1. Creating parameter distribution comparison...", end=' ')
        self._create_parameter_distributions(metrics)
        print("✓")
        
        print("2. Creating scalability analysis...", end=' ')
        self._create_scalability_plots(metrics)
        print("✓")
        
        print("3. Creating detailed scenario plots...", end=' ')
        self._create_detailed_scenarios(metrics)
        print("✓")
        
        print("4. Creating heatmap analysis...", end=' ')
        self._create_heatmaps(metrics)
        print("✓")
    
    def _create_parameter_distributions(self, metrics):
        """Create parameter distribution plots"""
        fig = plt.figure(figsize=(20, 12))
        
        n_scenarios = len(self.device_counts)
        
        # For each scenario, create distribution plots
        for idx, n_devices in enumerate(self.device_counts):
            data = self.combined_data[n_devices]
            
            # DR Distribution
            ax1 = plt.subplot(3, n_scenarios, idx + 1)
            ai_dr = self.ai_data[n_devices]['dr'].value_counts().sort_index()
            conv_dr = self.conv_data[n_devices]['dr'].value_counts().sort_index()
            
            x = np.arange(len(ai_dr))
            width = 0.35
            ax1.bar(x - width/2, ai_dr.values, width, label='AI-Powered', color='#2ecc71')
            ax1.bar(x + width/2, conv_dr.values, width, label='Conventional', color='#e74c3c')
            ax1.set_xlabel('Data Rate (DR)', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Number of Nodes', fontweight='bold', fontsize=9)
            ax1.set_title(f'{n_devices} Devices', fontsize=10, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(ai_dr.index)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # TX Power Distribution
            ax2 = plt.subplot(3, n_scenarios, n_scenarios + idx + 1)
            ai_power = self.ai_data[n_devices]['txPower'].value_counts().sort_index()
            conv_power = self.conv_data[n_devices]['txPower'].value_counts().sort_index()
            
            x = np.arange(len(ai_power))
            width = 0.35
            ax2.bar(x - width/2, ai_power.values, width, label='AI-Powered', color='#3498db')
            ax2.bar(x + width/2, conv_power.values, width, label='Conventional', color='#f39c12')
            ax2.set_xlabel('TX Power (dBm)', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Number of Nodes', fontweight='bold', fontsize=9)
            ax2.set_xticks(x)
            ax2.set_xticklabels(ai_power.index)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Frequency Distribution
            ax3 = plt.subplot(3, n_scenarios, 2*n_scenarios + idx + 1)
            ai_freq = self.ai_data[n_devices]['freq'].value_counts().sort_index()
            conv_freq = self.conv_data[n_devices]['freq'].value_counts().sort_index()
            
            # Convert to MHz for readability
            ai_freq_mhz = ai_freq.copy()
            ai_freq_mhz.index = [f"{f/1e6:.1f}" for f in ai_freq.index]
            conv_freq_mhz = conv_freq.copy()
            conv_freq_mhz.index = [f"{f/1e6:.1f}" for f in conv_freq.index]
            
            x = np.arange(len(ai_freq_mhz))
            width = 0.35
            ax3.bar(x - width/2, ai_freq_mhz.values, width, label='AI-Powered', color='#9b59b6')
            ax3.bar(x + width/2, conv_freq_mhz.values, width, label='Conventional', color='#e67e22')
            ax3.set_xlabel('Frequency (MHz)', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Number of Nodes', fontweight='bold', fontsize=9)
            ax3.set_xticks(x)
            ax3.set_xticklabels(ai_freq_mhz.index, rotation=45, ha='right')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scalability_plots(self, metrics):
        """Create scalability analysis"""
        fig = plt.figure(figsize=(18, 10))
        
        device_counts = self.device_counts
        
        # Extract data
        ai_avg_dr = [metrics[n]['AI-Powered']['Avg_DR'] for n in device_counts]
        conv_avg_dr = [metrics[n]['Conventional']['Avg_DR'] for n in device_counts]
        ai_avg_power = [metrics[n]['AI-Powered']['Avg_TxPower'] for n in device_counts]
        conv_avg_power = [metrics[n]['Conventional']['Avg_TxPower'] for n in device_counts]
        ai_high_dr = [metrics[n]['AI-Powered']['High_DR_Nodes'] for n in device_counts]
        conv_high_dr = [metrics[n]['Conventional']['High_DR_Nodes'] for n in device_counts]
        ai_low_power = [metrics[n]['AI-Powered']['Low_Power_Nodes'] for n in device_counts]
        conv_low_power = [metrics[n]['Conventional']['Low_Power_Nodes'] for n in device_counts]
        
        # 1. Average DR
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(device_counts, ai_avg_dr, marker='o', linewidth=2.5, markersize=8,
                label='AI-Powered', color='#2ecc71')
        ax1.plot(device_counts, conv_avg_dr, marker='s', linewidth=2.5, markersize=8,
                label='Conventional', color='#e74c3c')
        ax1.set_xlabel('Number of Devices', fontweight='bold')
        ax1.set_ylabel('Average Data Rate', fontweight='bold')
        ax1.set_title('Average DR Allocation', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Average TX Power
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(device_counts, ai_avg_power, marker='o', linewidth=2.5, markersize=8,
                label='AI-Powered', color='#3498db')
        ax2.plot(device_counts, conv_avg_power, marker='s', linewidth=2.5, markersize=8,
                label='Conventional', color='#f39c12')
        ax2.set_xlabel('Number of Devices', fontweight='bold')
        ax2.set_ylabel('Average TX Power (dBm)', fontweight='bold')
        ax2.set_title('Average Power Allocation', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. High DR Nodes
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(device_counts, ai_high_dr, marker='o', linewidth=2.5, markersize=8,
                label='AI-Powered', color='#2ecc71')
        ax3.plot(device_counts, conv_high_dr, marker='s', linewidth=2.5, markersize=8,
                label='Conventional', color='#e74c3c')
        ax3.set_xlabel('Number of Devices', fontweight='bold')
        ax3.set_ylabel('Nodes with High DR (≥4)', fontweight='bold')
        ax3.set_title('High Data Rate Usage', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Low Power Nodes
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(device_counts, ai_low_power, marker='o', linewidth=2.5, markersize=8,
                label='AI-Powered', color='#3498db')
        ax4.plot(device_counts, conv_low_power, marker='s', linewidth=2.5, markersize=8,
                label='Conventional', color='#f39c12')
        ax4.set_xlabel('Number of Devices', fontweight='bold')
        ax4.set_ylabel('Nodes with Low Power (≤10dBm)', fontweight='bold')
        ax4.set_title('Energy-Efficient Allocation', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. DR vs Power Trade-off
        ax5 = plt.subplot(2, 3, 5)
        for n_devices in device_counts:
            ai_data = self.ai_data[n_devices]
            ax5.scatter(ai_data['dr'], ai_data['txPower'], alpha=0.3, s=20, label=f'{n_devices} nodes')
        ax5.set_xlabel('Data Rate (DR)', fontweight='bold')
        ax5.set_ylabel('TX Power (dBm)', fontweight='bold')
        ax5.set_title('AI-Powered: DR vs Power Trade-off', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Conventional DR vs Power Trade-off
        ax6 = plt.subplot(2, 3, 6)
        for n_devices in device_counts:
            conv_data = self.conv_data[n_devices]
            ax6.scatter(conv_data['dr'], conv_data['txPower'], alpha=0.3, s=20, label=f'{n_devices} nodes')
        ax6.set_xlabel('Data Rate (DR)', fontweight='bold')
        ax6.set_ylabel('TX Power (dBm)', fontweight='bold')
        ax6.set_title('Conventional: DR vs Power Trade-off', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_scenarios(self, metrics):
        """Create detailed per-scenario analysis"""
        n_scenarios = len(self.device_counts)
        fig = plt.figure(figsize=(16, 5 * n_scenarios))
        
        plot_idx = 1
        for n_devices in self.device_counts:
            ai_data = self.ai_data[n_devices]
            conv_data = self.conv_data[n_devices]
            
            # DR Box Plot
            ax1 = plt.subplot(n_scenarios, 3, plot_idx)
            data_combined = pd.concat([
                ai_data[['dr', 'ADR_Type']],
                conv_data[['dr', 'ADR_Type']]
            ], ignore_index=True)
            sns.boxplot(data=data_combined, x='ADR_Type', y='dr', ax=ax1, hue='ADR_Type',
                       palette={'AI-Powered': '#2ecc71', 'Conventional': '#e74c3c'}, legend=False)
            ax1.set_title(f'{n_devices} Devices - DR Distribution', fontweight='bold')
            ax1.set_ylabel('Data Rate', fontweight='bold')
            ax1.set_xlabel('')
            
            # Power Box Plot
            ax2 = plt.subplot(n_scenarios, 3, plot_idx + 1)
            power_combined = pd.concat([
                ai_data[['txPower', 'ADR_Type']],
                conv_data[['txPower', 'ADR_Type']]
            ], ignore_index=True)
            sns.boxplot(data=power_combined, x='ADR_Type', y='txPower', ax=ax2, hue='ADR_Type',
                       palette={'AI-Powered': '#3498db', 'Conventional': '#f39c12'}, legend=False)
            ax2.set_title(f'{n_devices} Devices - Power Distribution', fontweight='bold')
            ax2.set_ylabel('TX Power (dBm)', fontweight='bold')
            ax2.set_xlabel('')
            
            # DR-Power Joint Distribution
            ax3 = plt.subplot(n_scenarios, 3, plot_idx + 2)
            ax3.scatter(ai_data['dr'], ai_data['txPower'], alpha=0.5, s=30,
                       label='AI-Powered', color='#2ecc71')
            ax3.scatter(conv_data['dr'], conv_data['txPower'], alpha=0.5, s=30,
                       label='Conventional', color='#e74c3c')
            ax3.set_xlabel('Data Rate', fontweight='bold')
            ax3.set_ylabel('TX Power (dBm)', fontweight='bold')
            ax3.set_title(f'{n_devices} Devices - Parameter Joint Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plot_idx += 3
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_scenarios.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_heatmaps(self, metrics):
        """Create heatmap comparisons"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        device_counts = self.device_counts
        
        # Average DR Heatmap
        dr_data = np.array([[metrics[n]['AI-Powered']['Avg_DR'],
                            metrics[n]['Conventional']['Avg_DR']] for n in device_counts])
        sns.heatmap(dr_data.T, annot=True, fmt='.2f', cmap='YlGnBu',
                   xticklabels=device_counts, yticklabels=['AI-Powered', 'Conventional'],
                   ax=axes[0,0], cbar_kws={'label': 'Average DR'})
        axes[0,0].set_title('Average Data Rate', fontweight='bold')
        axes[0,0].set_xlabel('Number of Devices', fontweight='bold')
        
        # Average TX Power Heatmap
        power_data = np.array([[metrics[n]['AI-Powered']['Avg_TxPower'],
                               metrics[n]['Conventional']['Avg_TxPower']] for n in device_counts])
        sns.heatmap(power_data.T, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=device_counts, yticklabels=['AI-Powered', 'Conventional'],
                   ax=axes[0,1], cbar_kws={'label': 'TX Power (dBm)'})
        axes[0,1].set_title('Average TX Power', fontweight='bold')
        axes[0,1].set_xlabel('Number of Devices', fontweight='bold')
        
        # High DR Nodes Heatmap
        high_dr_data = np.array([[metrics[n]['AI-Powered']['High_DR_Nodes'],
                                 metrics[n]['Conventional']['High_DR_Nodes']] for n in device_counts])
        sns.heatmap(high_dr_data.T, annot=True, fmt='d', cmap='Greens',
                   xticklabels=device_counts, yticklabels=['AI-Powered', 'Conventional'],
                   ax=axes[1,0], cbar_kws={'label': 'Nodes'})
        axes[1,0].set_title('High DR Nodes (DR≥4)', fontweight='bold')
        axes[1,0].set_xlabel('Number of Devices', fontweight='bold')
        
        # Low Power Nodes Heatmap
        low_power_data = np.array([[metrics[n]['AI-Powered']['Low_Power_Nodes'],
                                   metrics[n]['Conventional']['Low_Power_Nodes']] for n in device_counts])
        sns.heatmap(low_power_data.T, annot=True, fmt='d', cmap='Blues',
                   xticklabels=device_counts, yticklabels=['AI-Powered', 'Conventional'],
                   ax=axes[1,1], cbar_kws={'label': 'Nodes'})
        axes[1,1].set_title('Low Power Nodes (≤10dBm)', fontweight='bold')
        axes[1,1].set_xlabel('Number of Devices', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'heatmap_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    print("Checking file paths...")
    
    # Verify files exist
    all_files = {**AI_FILES, **CONVENTIONAL_FILES}
    missing = []
    
    for n, path in all_files.items():
        if not os.path.exists(path):
            missing.append(path)
            print(f"  ✗ Not found: {path}")
        else:
            print(f"  ✓ Found: {path}")
    
    if missing:
        print(f"\n❌ Error: {len(missing)} file(s) not found!")
        return
    
    print("\n✓ All files found!\n")
    
    # Get common device counts
    common = set(AI_FILES.keys()) & set(CONVENTIONAL_FILES.keys())
    if not common:
        print("❌ Error: No matching device counts!")
        return
    
    ai_files = {k: v for k, v in AI_FILES.items() if k in common}
    conv_files = {k: v for k, v in CONVENTIONAL_FILES.items() if k in common}
    
    print(f"Analyzing {len(common)} scenarios: {sorted(common)}\n")
    
    # Run comparison
    try:
        comparison = ParameterComparison(ai_files, conv_files, OUTPUT_DIR)
        metrics = comparison.calculate_metrics()
        comparison.print_summary_table(metrics)
        comparison.create_visualizations(metrics)
        
        print("\n" + "="*100)
        print("✓ ANALYSIS COMPLETE!")
        print("="*100)
        print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
        print("\nGenerated files:")
        print("  1. parameter_distributions.png - DR, Power, Frequency distributions")
        print("  2. scalability_analysis.png - Parameter trends across scenarios")
        print("  3. detailed_scenarios.png - Per-scenario box plots and scatter")
        print("  4. heatmap_comparison.png - Parameter allocation heatmaps")
        print("="*100)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()