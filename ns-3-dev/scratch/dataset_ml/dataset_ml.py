#!/usr/bin/env python3
"""
LoRaWAN Network-Aware Config Predictor - MINIMAL VERSION
Just 2 functions: predict_best() and process_all_nodes()
"""

import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('lorawan_adr_model_b_energy.pkl')
p = joblib.load('path_loss_params_energy.pkl')

DR_TO_SF = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
SF_TO_DR = {12: 0, 11: 1, 10: 2, 9: 3, 8: 4, 7: 5}

def predict_best(distance, freq_usage={868.1: 0, 868.3: 0, 868.5: 0}, total_nodes=1):
    """
    Predict best config for one node (network-aware)
    
    Args:
        distance: Distance to gateway (m)
        freq_usage: Dict of current frequency usage
        total_nodes: Total nodes in network
    
    Returns:
        Best config dict
    """
    best = {'score': -1}
    
    for dr in range(6):
        for freq in [868.1, 868.3, 868.5]:
            for power in [6, 8, 10, 12, 14]:
                # Calculate link quality
                pl = p['PL0'] + 10 * p['n'] * np.log10(distance)
                rssi = power - pl
                margin = rssi - p['sensitivity_map'][dr]
                
                if margin < 2: continue  # Safety filter
                
                energy = p['energy_lookup'][dr][power]
                
                # Predict PDR
                X = pd.DataFrame([{
                    'distance': distance, 'distance_normalized': distance/6000,
                    'radial_distance': distance, 'dr': dr, 'freq': freq,
                    'txPower': power, 'RSSI': rssi, 'SNR': rssi - p['noise_floor_dBm'],
                    'Sensitivity': p['sensitivity_map'][dr], 'link_margin_dB': margin,
                    'snr_margin_dB': rssi - p['noise_floor_dBm'] + 7.5,
                    'is_above_sensitivity': 1, 'power_distance_ratio': power/(distance+1),
                    'EnergyPerPacket_mJ': energy
                }])
                
                pdr = model.predict(X)[0]
                
                # Multi-objective score: PDR + Energy
                base_score = 0.6 * (pdr/100) + 0.4 * (1 - energy/3.95)
                
                # Frequency penalty (network-aware)
                freq_penalty = (freq_usage.get(freq, 0) / total_nodes) * 0.15
                final_score = base_score * (1 - freq_penalty)
                
                if final_score > best['score']:
                    best = {
                        'SF': 12-dr, 'DR':SF_TO_DR[12-dr], 'Freq': freq, 'Power': power,
                        'PDR': round(pdr, 1), 'Energy': round(energy, 3),
                        'Battery': round(21.6/(energy/1000)/100/365, 2),
                        'Margin': round(margin, 1), 'score': final_score
                    }
    
    return best if best['score'] > 0 else None

def process_all_nodes(distances):
    """
    Process all nodes with network-aware frequency allocation
    
    Args:
        distances: List of distances
    
    Returns:
        List of configs
    """

    freq_usage = {868.1: 0, 868.3: 0, 868.5: 0}
    configs = []
    
    for i, dist in enumerate(distances):
        # Get best config considering current frequency usage
        cfg = predict_best(dist, freq_usage, len(distances))
        
        if cfg:
            # Update frequency usage
            freq_usage[cfg['Freq']] += 1
            configs.append(cfg)
            print(f"Node {i:2d} ({dist:6.1f}m): SF{cfg['SF']}, DR{SF_TO_DR[cfg['SF']]}, {cfg['Freq']} MHz, "
                  f"{cfg['Power']}dBm → PDR:{cfg['PDR']}%, E:{cfg['Energy']}mJ")
        else:
            configs.append(None)
            print(f"Node {i:2d} ({dist:6.1f}m): ❌ No safe config")
    
    # Show frequency distribution
    print("\n" + "="*60)
    print("📊 Frequency Distribution:")
    total = sum(freq_usage.values())
    for freq, count in sorted(freq_usage.items()):
        pct = count/total*100 if total > 0 else 0
        bar = "█" * int(pct/3)
        print(f"  {freq} MHz: {count:2d} nodes ({pct:5.1f}%) {bar}")
    
    print(f"\nTotal: {len([c for c in configs if c])}/{len(distances)} nodes assigned")
    print("="*60)
    
    return configs


def generate_positions(num_nodes, radius_meters):
    np.random.seed(42) 
    # Generate random positions in a disc
    positions = []
    list_distance = []
    for i in range(num_nodes):
        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Random radius (uniform distribution in disc)
        r = radius_meters * np.sqrt(np.random.uniform(0, 1))
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0  # Fixed height for end devices
        
        # Calculate distance from gateway at (0, 0, 15)
        distance = np.sqrt(x**2 + y**2 + (z - 15)**2)
        
        positions.append({
            'nodeIndex': i,
            'x': x,
            'y': y,
            'z': z,
            'distance': distance
        })

        list_distance.append(distance.tolist())
    
    # print(positions)
    nama_file = f'node_positions_{num_nodes}_{radius_meters}.csv'
    df = pd.DataFrame(positions)
    df.to_csv(nama_file, index=False)
    print(f"✓ Saved position file for NS-3 to: {nama_file}")

    return positions, list_distance


def ml_predict(num_nodes, radius_meters):
    print(f"\n🌐 Processing {num_nodes} in {radius_meters} meter nodes with Frequency Aware")
    print("="*60)
    params_values = []
    positions_params = generate_positions(num_nodes, radius_meters)
    params_ns3 = process_all_nodes(positions_params[1])

    nodeId = 0
    for i in params_ns3:
        params_values.append({
                'experiment': 1,
                'nodeId': nodeId,
                'combId': 0,
                'dr': i['DR'],
                'freq': int(i['Freq']*1000000),
                'txPower': i['Power']
        })
        nodeId = nodeId + 1

    nama_file = f'experiment_schedule_{num_nodes}_{radius_meters}.csv'
    # print(params_values)
    df = pd.DataFrame(params_values)
    df.to_csv(nama_file, index=False)


    return print(f"✓ Saved Postion to: node_positions.csv & parameters to: experiment_schedule.csv")

# ========== USAGE ==========
ml_predict(20,5000)
# ml_predict(400,5000)
# ml_predict(600,5000)
# ml_predict(800,5000)
# ml_predict(1000,5000)