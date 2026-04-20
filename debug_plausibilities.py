import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    graph_path = os.path.join('data', 'index', 'plausibility_graph.pkl')
    out_dir = os.path.join('results', 'plausibilities')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(graph_path):
        print(f"Graph file not found at {graph_path}.")
        return

    print("Loading plausibility graph...")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    n_regions = 256
    
    # Initialize a 256x256 matrix with NaNs (which will be colored gray)
    heatmap_data = np.full((n_regions, n_regions), np.nan)
    
    for u, neighbors in graph.items():
        if 0 <= u < n_regions:
            for v, cost in neighbors.items():
                if 0 <= v < n_regions:
                    heatmap_data[u, v] = cost
                    
    print("Generating the heatmap...")
    # Set up the matplotlib figure for a professional look
    plt.figure(figsize=(10, 8), dpi=300)
    
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color='gray')
    
    # Plot the heatmap
    im = plt.imshow(heatmap_data, cmap=cmap, aspect='equal', origin='upper',
                    interpolation='nearest', vmin=np.nanmin(heatmap_data), vmax=np.nanmax(heatmap_data))
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Transition Cost (Lower is more plausible)', rotation=270, labelpad=15)
    
    # Add labels and formatting
    plt.title('Codebook Region Pairwise Plausibility', fontsize=16, pad=15)
    plt.xlabel('Target Region', fontsize=12)
    plt.ylabel('Source Region', fontsize=12)
    
    # Ticks formatting
    plt.xticks(np.arange(0, n_regions+1, 32))
    plt.yticks(np.arange(0, n_regions+1, 32))
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'plausibility_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Professional heatmap saved to {out_path}")

if __name__ == '__main__':
    main()