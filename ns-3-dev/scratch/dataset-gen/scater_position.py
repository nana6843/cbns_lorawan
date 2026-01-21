# import pandas as pd
# import matplotlib.pyplot as plt

# file = "/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset-gen/node_positions.csv"
# df = pd.read_csv(file)


# plt.figure(figsize=(8,7))
# scatter = plt.scatter(df["x"], df["y"], c=df["distance"], cmap="viridis", s=30)

# plt.scatter(0, 0, marker='x', color='red', s=120, label='Gateway (0,0)')

# plt.colorbar(scatter, label="Distance to Gateway (m)")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# plt.title("Spatial Distribution of Nodes by Distance")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# # SAVE IMAGE
# plt.savefig("/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset-gen/node_distance_colormap.png", dpi=300)

# plt.show()


from graphviz import Digraph

dot = Digraph(comment='Dataset Generation Flow', format='png')
dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', bgcolor='white')

# Node style
common = {'shape': 'box', 'style':'rounded,filled', 'color':'#0A3D91', 'fillcolor':'#E6EEF8', 'fontname':'Arial', 'fontsize':'11'}

dot.node('P','Python Processing\n(generate_allocation.py\n& generate_positions.py)', **common)
dot.node('I1','Input 1:\nnode_positions.csv\n(500 ED + normal distribution)', **common)
dot.node('I2','Input 2:\nexperiment_schedule.csv\n(500 ED with 3 Freq × 6 DR × 5 Tx Power)', **common)
dot.node('NS','NS-3 (dataset-gen.cc)\n90 experiments with random parameters', **common)
dot.node('O','Output dataset\n45,000 datapoints (13 columns)', **common)
dot.node('PR','Python Processing\n(dataset_rssi_snr.py)', **common)
dot.node('F','Final Dataset\n45,000 datapoints (24 columns)', **common)
dot.node('ML','NEXT PROCESS:\nMachine Learning', **common)

# Arrows
dot.edge('P','I1', color='#0A3D91')
dot.edge('P','I2', color='#0A3D91')
dot.edge('I1','NS', color='#0A3D91')
dot.edge('I2','NS', color='#0A3D91')
dot.edge('NS','O', color='#0A3D91')
dot.edge('O','PR', color='#0A3D91')
dot.edge('PR','F', color='#0A3D91')
dot.edge('F','ML', color='#0A3D91')

path = '/home/nru/CBNS_NEW2/ns-3-dev/scratch/dataset-gen/dataset_flow_styled.png'
dot.render(path, view=False)
path
