#!/usr/bin/env python3
"""
Generate fixed positions for all 500 nodes
To be used consistently across all 90 experiments
"""

import numpy as np
import pandas as pd

NUM_NODES = 500
RADIUS_METERS = 5000

# Generate random positions in a disc
positions = []
for i in range(NUM_NODES):
    # Random angle
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Random radius (uniform distribution in disc)
    r = RADIUS_METERS * np.sqrt(np.random.uniform(0, 1))
    
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

nama_file = f'node_positions_nDevices_{NUM_NODES}_Radius_{RADIUS_METERS}.csv'
df = pd.DataFrame(positions)
df.to_csv(nama_file, index=False)

print(f"✓ Generated {NUM_NODES} fixed positions")
print(f"Distance range: {df['distance'].min():.2f}m - {df['distance'].max():.2f}m")
print(f"✓ Saved to: {nama_file}")