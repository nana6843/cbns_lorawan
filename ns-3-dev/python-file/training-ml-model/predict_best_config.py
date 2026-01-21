#!/usr/bin/env python3
"""
LoRaWAN ML-ADR - Best Configuration Predictor (ENERGY-AWARE)
Simple script to predict best DR, Freq, TxPower for a node
Multi-Objective: PDR + Energy Efficiency
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# ========== CHECK FILES ==========

print("="*70)
print("LoRaWAN Best Config Predictor (Model B - Energy-Aware)")
print("="*70)

# Check if model files exist
required_files = [
    'lorawan_adr_model_b_energy.pkl',
    'path_loss_params_energy.pkl'
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"\n❌ ERROR: Missing required files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\nPlease run training script first: python ml_ns3_predictor.py")
    sys.exit(1)

# ========== LOAD MODEL ==========

try:
    model = joblib.load('lorawan_adr_model_b_energy.pkl')
    params = joblib.load('path_loss_params_energy.pkl')
    print("✓ Loaded Energy-Aware Model B successfully")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# Extract parameters
PL0 = params['PL0']
n = params['n']
NOISE_FLOOR_DBM = params['noise_floor_dBm']
SENSITIVITY_MAP = params['sensitivity_map']
TOA_LOOKUP = params['toa_lookup']
ENERGY_LOOKUP = params['energy_lookup']  # ⭐ NEW: Energy lookup table

print(f"✓ Path Loss Model: PL0={PL0:.2f} dB, n={n:.2f}")
print(f"✓ Energy Lookup Table loaded")

# ========== CONFIGURATION ==========

# Available options
DR_OPTIONS = [0, 1, 2, 3, 4, 5]  # DR0=SF12, DR5=SF7
FREQ_OPTIONS = [868.1, 868.3, 868.5]  # MHz
POWER_OPTIONS = [6, 8, 10, 12, 14]  # dBm

# DR to SF mapping
DR_TO_SF = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}

# Max distance for normalization
MAX_DISTANCE = 6000  # meters

# Battery parameters for lifetime estimation
BATTERY_CAPACITY_J = 21.6  # 2000mAh @ 3V
PACKETS_PER_DAY = 100

# ========== HELPER FUNCTIONS ==========

def path_loss_model(distance):
    """Calculate path loss using log-distance model"""
    d0 = 1.0
    return PL0 + 10 * n * np.log10(distance / d0)

def calculate_link_quality(distance, txPower, dr):
    """
    Calculate RSSI, SNR, Sensitivity, Link Margin
    
    Args:
        distance: Distance to gateway (meters)
        txPower: Transmission power (dBm)
        dr: Data Rate (0-5)
        
    Returns:
        tuple: (rssi, snr, sensitivity, link_margin, snr_margin, is_above)
    """
    # Path loss
    path_loss = path_loss_model(distance)
    
    # RSSI
    rssi = txPower - path_loss
    
    # SNR
    snr = rssi - NOISE_FLOOR_DBM
    
    # Sensitivity
    sensitivity = SENSITIVITY_MAP.get(dr, -130.0)
    
    # Margins
    link_margin = rssi - sensitivity
    snr_margin = snr - (-7.5)
    is_above = 1 if rssi > sensitivity else 0
    
    return rssi, snr, sensitivity, link_margin, snr_margin, is_above

def get_energy_consumption(dr, power):
    """
    Get energy consumption from lookup table
    
    Args:
        dr: Data Rate (0-5)
        power: Transmission power (dBm)
        
    Returns:
        float: Energy per packet (mJ)
    """
    return ENERGY_LOOKUP[dr][power]

def calculate_battery_lifetime(energy_mJ, packets_per_day=100):
    """
    Calculate battery lifetime in years
    
    Args:
        energy_mJ: Energy per packet (mJ)
        packets_per_day: Number of packets per day
        
    Returns:
        float: Battery lifetime in years
    """
    energy_J = energy_mJ / 1000.0
    total_packets = BATTERY_CAPACITY_J / energy_J
    days = total_packets / packets_per_day
    years = days / 365
    return years

def predict_pdr(distance, dr, freq, power):
    """
    Predict PDR for given configuration (with energy awareness)
    
    Args:
        distance: Distance to gateway (meters)
        dr: Data Rate (0-5)
        freq: Frequency (MHz)
        power: Transmission power (dBm)
        
    Returns:
        tuple: (predicted_pdr, rssi, link_margin, is_above, energy_mJ)
    """
    # Calculate link quality
    rssi, snr, sensitivity, link_margin, snr_margin, is_above = \
        calculate_link_quality(distance, power, dr)
    
    # Get energy consumption
    energy_mJ = get_energy_consumption(dr, power)
    
    # Prepare features
    distance_norm = distance / MAX_DISTANCE
    radial = distance
    power_dist_ratio = power / (distance + 1)
    
    # Feature vector (14 features for Model B with Energy)
    X = pd.DataFrame([{
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
        'EnergyPerPacket_mJ': energy_mJ  # ⭐ Energy feature
    }])
    
    # Predict
    pred_pdr = model.predict(X)[0]
    
    # Clamp to [0, 100]
    pred_pdr = max(0, min(100, pred_pdr))
    
    return pred_pdr, rssi, link_margin, is_above, energy_mJ

def find_best_config(distance, safety_margin_dB=3, pdr_threshold=80, top_n=5,
                     weight_pdr=0.6, weight_energy=0.4):
    """
    Find best configuration for given distance (ENERGY-AWARE)
    
    Args:
        distance: Distance to gateway (meters)
        safety_margin_dB: Minimum link margin (dB)
        pdr_threshold: Minimum acceptable PDR (%)
        top_n: Number of top configs to return
        weight_pdr: Weight for PDR in multi-objective score (default: 0.6)
        weight_energy: Weight for energy efficiency in multi-objective score (default: 0.4)
        
    Returns:
        DataFrame: Top N configurations
    """
    print(f"\n{'='*70}")
    print(f"Finding Best Energy-Efficient Config for Node at {distance:.0f}m")
    print(f"Safety Margin: {safety_margin_dB} dB | PDR Threshold: {pdr_threshold}%")
    print(f"Weights: PDR={weight_pdr:.0%}, Energy={weight_energy:.0%}")
    print(f"{'='*70}")
    
    results = []
    
    # Calculate max energy for normalization
    max_energy_mJ = max([ENERGY_LOOKUP[dr][14] for dr in DR_OPTIONS])
    
    # Test all possible configurations
    total_configs = len(DR_OPTIONS) * len(FREQ_OPTIONS) * len(POWER_OPTIONS)
    print(f"\nTesting {total_configs} configurations...")
    
    for dr in DR_OPTIONS:
        for freq in FREQ_OPTIONS:
            for power in POWER_OPTIONS:
                
                # Predict PDR and get link quality + energy
                pred_pdr, rssi, link_margin, is_above, energy_mJ = \
                    predict_pdr(distance, dr, freq, power)
                
                # Apply safety margin filter
                if link_margin < safety_margin_dB:
                    continue
                
                # Get time-on-air (for display)
                toa = TOA_LOOKUP[dr]
                
                # Calculate battery lifetime
                battery_years = calculate_battery_lifetime(energy_mJ, PACKETS_PER_DAY)
                
                # ⭐ ENERGY-BASED MULTI-OBJECTIVE SCORE
                if pred_pdr >= pdr_threshold:
                    pdr_score = pred_pdr / 100.0
                    energy_score = 1 - (energy_mJ / max_energy_mJ)
                    
                    # Multi-objective: PDR + Energy Efficiency
                    score = weight_pdr * pdr_score + weight_energy * energy_score
                else:
                    score = 0
                
                # Get SF
                sf = DR_TO_SF[dr]
                
                # Safety level
                if link_margin >= 10:
                    safety = "🟢 High"
                elif link_margin >= 5:
                    safety = "🟡 Medium"
                else:
                    safety = "🟠 Low"
                
                results.append({
                    'DR': dr,
                    'SF': sf,
                    'Freq_MHz': freq,
                    'TxPower_dBm': power,
                    'Predicted_PDR_%': round(pred_pdr, 1),
                    'Energy_mJ': round(energy_mJ, 3),
                    'Battery_Years': round(battery_years, 2),
                    'ToA_ms': round(toa, 1),
                    'RSSI_dBm': round(rssi, 1),
                    'LinkMargin_dB': round(link_margin, 1),
                    'Safety': safety,
                    'Score': round(score, 4)
                })
    
    # Check if any configs found
    if len(results) == 0:
        print(f"\n⚠️  No configs meet safety margin of {safety_margin_dB} dB!")
        
        if safety_margin_dB > 1:
            print(f"   Retrying with relaxed margin (1 dB)...")
            return find_best_config(distance, safety_margin_dB=1, 
                                   pdr_threshold=pdr_threshold, top_n=top_n,
                                   weight_pdr=weight_pdr, weight_energy=weight_energy)
        else:
            print(f"\n❌ Distance too large!")
            print(f"   Suggestions:")
            print(f"   - Reduce distance to gateway")
            print(f"   - Add more gateways")
            print(f"   - Use higher TxPower if possible")
            return pd.DataFrame()
    
    # Create DataFrame and sort by multi-objective score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Score', ascending=False)
    
    print(f"✓ Found {len(df_results)} safe configurations\n")
    
    # Display top N
    print(f"Top {min(top_n, len(df_results))} Energy-Optimized Configurations:")
    print("="*70)
    
    display_cols = ['SF', 'Freq_MHz', 'TxPower_dBm', 'Predicted_PDR_%', 
                    'Energy_mJ', 'Battery_Years', 'LinkMargin_dB', 'Safety']
    print(df_results[display_cols].head(top_n).to_string(index=False))
    
    # Show best config details
    best = df_results.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"🎯 RECOMMENDED ENERGY-EFFICIENT CONFIGURATION:")
    print(f"{'='*70}")
    print(f"  Spreading Factor:  SF{best['SF']}")
    print(f"  Frequency:         {best['Freq_MHz']} MHz")
    print(f"  Transmission Power: {best['TxPower_dBm']} dBm")
    print(f"\n  Expected Performance:")
    print(f"  ├─ PDR:            {best['Predicted_PDR_%']:.1f}%")
    print(f"  ├─ Energy/Packet:  {best['Energy_mJ']:.3f} mJ")
    print(f"  ├─ Battery Life:   {best['Battery_Years']:.1f} years (@ 100 pkt/day)")
    print(f"  ├─ Time-on-Air:    {best['ToA_ms']:.1f} ms")
    print(f"  ├─ RSSI:           {best['RSSI_dBm']:.1f} dBm")
    print(f"  ├─ Link Margin:    {best['LinkMargin_dB']:.1f} dB")
    print(f"  └─ Safety Level:   {best['Safety']}")
    print(f"{'='*70}")
    
    # Energy savings comparison
    if len(df_results) > 1:
        worst_energy = df_results['Energy_mJ'].max()
        best_energy = best['Energy_mJ']
        energy_saving = ((worst_energy - best_energy) / worst_energy) * 100
        print(f"\n💡 Energy Optimization:")
        print(f"   Best config saves up to {energy_saving:.1f}% energy vs worst safe config")
        print(f"   Battery lifetime: {best['Battery_Years']:.1f} years")
    
    return df_results.head(top_n)

# ========== INTERACTIVE MODE ==========

def interactive_mode():
    """Interactive CLI for single queries"""
    print(f"\n{'='*70}")
    print("Interactive Mode - Energy-Aware Config Predictor")
    print("="*70)
    print("Enter distance to get best energy-efficient configuration")
    print("Type 'q' to quit")
    print("="*70)
    
    while True:
        print("\n")
        user_input = input("Enter node distance (meters) [or 'q' to quit]: ").strip()
        
        if user_input.lower() == 'q':
            print("Goodbye! 👋")
            break
        
        try:
            distance = float(user_input)
            
            if distance <= 0:
                print("❌ Distance must be positive!")
                continue
            
            if distance > 7000:
                print("⚠️  Warning: Distance very large! Results may be unreliable.")
            
            # Get safety margin (optional)
            margin_input = input("Safety margin in dB [default: 3]: ").strip()
            safety_margin = float(margin_input) if margin_input else 3.0
            
            if safety_margin < 0:
                print("❌ Safety margin must be non-negative!")
                continue
            
            # Get optimization weights (optional)
            print("\nOptimization weights (must sum to 1.0):")
            weight_pdr_input = input("  PDR weight [default: 0.6]: ").strip()
            weight_pdr = float(weight_pdr_input) if weight_pdr_input else 0.6
            
            weight_energy = 1.0 - weight_pdr
            print(f"  Energy weight: {weight_energy:.1f} (auto-calculated)")
            
            if weight_pdr < 0 or weight_pdr > 1:
                print("❌ Weight must be between 0 and 1!")
                continue
            
            # Find best config
            find_best_config(distance, safety_margin_dB=safety_margin, 
                           pdr_threshold=80, top_n=3,
                           weight_pdr=weight_pdr, weight_energy=weight_energy)
            
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
        except Exception as e:
            print(f"❌ Error: {e}")

# ========== BATCH MODE ==========

def batch_mode(distances, output_file='predictions_energy.csv', 
               weight_pdr=0.6, weight_energy=0.4):
    """Process multiple distances and save to CSV"""
    print(f"\n{'='*70}")
    print(f"Batch Mode - Processing {len(distances)} distances")
    print(f"Weights: PDR={weight_pdr:.0%}, Energy={weight_energy:.0%}")
    print("="*70)
    
    all_results = []
    
    for i, dist in enumerate(distances, 1):
        print(f"\n[{i}/{len(distances)}] Processing distance: {dist}m")
        
        configs = find_best_config(dist, safety_margin_dB=3, 
                                   pdr_threshold=80, top_n=1,
                                   weight_pdr=weight_pdr, 
                                   weight_energy=weight_energy)
        
        if len(configs) > 0:
            best = configs.iloc[0]
            all_results.append({
                'Distance_m': dist,
                'Recommended_SF': best['SF'],
                'Recommended_DR': best['DR'],
                'Recommended_Freq_MHz': best['Freq_MHz'],
                'Recommended_TxPower_dBm': best['TxPower_dBm'],
                'Predicted_PDR_%': best['Predicted_PDR_%'],
                'Energy_mJ': best['Energy_mJ'],
                'Battery_Lifetime_Years': best['Battery_Years'],
                'ToA_ms': best['ToA_ms'],
                'RSSI_dBm': best['RSSI_dBm'],
                'LinkMargin_dB': best['LinkMargin_dB']
            })
        else:
            all_results.append({
                'Distance_m': dist,
                'Recommended_SF': 'N/A',
                'Recommended_DR': 'N/A',
                'Recommended_Freq_MHz': 'N/A',
                'Recommended_TxPower_dBm': 'N/A',
                'Predicted_PDR_%': 0,
                'Energy_mJ': 0,
                'Battery_Lifetime_Years': 0,
                'ToA_ms': 0,
                'RSSI_dBm': 0,
                'LinkMargin_dB': 0
            })
    
    # Save to CSV
    df_batch = pd.DataFrame(all_results)
    df_batch.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Batch processing complete!")
    print(f"✅ Saved {len(df_batch)} results to: {output_file}")
    print(f"{'='*70}")
    
    # Show summary
    valid_results = df_batch[df_batch['Predicted_PDR_%'] > 0]
    if len(valid_results) > 0:
        print(f"\nSummary:")
        print(f"  Average PDR:           {valid_results['Predicted_PDR_%'].mean():.1f}%")
        print(f"  Average Energy:        {valid_results['Energy_mJ'].mean():.3f} mJ/packet")
        print(f"  Average Battery Life:  {valid_results['Battery_Lifetime_Years'].mean():.1f} years")
        print(f"  Average Link Margin:   {valid_results['LinkMargin_dB'].mean():.1f} dB")
    
    return df_batch

# ========== MAIN ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LoRaWAN Energy-Aware Config Predictor using Model B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python predict_best_config.py --interactive
  
  # Single distance query
  python predict_best_config.py --distance 3500
  
  # With custom safety margin
  python predict_best_config.py --distance 5000 --margin 5
  
  # With custom optimization weights
  python predict_best_config.py --distance 3000 --weight-pdr 0.7 --weight-energy 0.3
  
  # Batch mode
  python predict_best_config.py --batch 1000 2000 3000 4000 5000
  
  # Batch with output file and custom weights
  python predict_best_config.py --batch 1000 2500 4000 --output my_results.csv --weight-pdr 0.5
        """
    )
    
    parser.add_argument('--distance', type=float, 
                       help='Node distance in meters')
    parser.add_argument('--margin', type=float, default=3.0,
                       help='Safety margin in dB (default: 3)')
    parser.add_argument('--weight-pdr', type=float, default=0.6,
                       help='Weight for PDR in multi-objective score (default: 0.6)')
    parser.add_argument('--weight-energy', type=float, default=0.4,
                       help='Weight for energy in multi-objective score (default: 0.4)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--batch', nargs='+', type=float,
                       help='Batch mode with multiple distances')
    parser.add_argument('--output', type=str, default='predictions_energy.csv',
                       help='Output file for batch mode (default: predictions_energy.csv)')
    
    args = parser.parse_args()
    
    # Validate weights
    if args.weight_pdr + args.weight_energy != 1.0:
        print(f"⚠️  Warning: Weights don't sum to 1.0. Auto-adjusting...")
        total = args.weight_pdr + args.weight_energy
        args.weight_pdr = args.weight_pdr / total
        args.weight_energy = args.weight_energy / total
        print(f"   Adjusted: PDR={args.weight_pdr:.2f}, Energy={args.weight_energy:.2f}")
    
    # Determine mode
    if args.interactive:
        # Interactive mode
        interactive_mode()
        
    elif args.batch:
        # Batch mode
        batch_mode(args.batch, output_file=args.output,
                  weight_pdr=args.weight_pdr, weight_energy=args.weight_energy)
        
    elif args.distance:
        # Single query
        find_best_config(args.distance, safety_margin_dB=args.margin,
                        pdr_threshold=80, top_n=5,
                        weight_pdr=args.weight_pdr, 
                        weight_energy=args.weight_energy)
        
    else:
        # Default: show examples
        print("\n" + "="*70)
        print("Example: Testing Multiple Distances with Energy Optimization")
        print("="*70)
        
        example_distances = [1000, 2500, 4000, 5500]
        
        for dist in example_distances:
            find_best_config(dist, safety_margin_dB=3, pdr_threshold=80, top_n=3,
                           weight_pdr=0.6, weight_energy=0.4)
        
        print("\n" + "="*70)
        print("💡 TIP: Use --help to see all available options")
        print("="*70)