import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

def create_plausibilities(args):
    # 1. Load Data
    print("Loading indexed data...")
    index_path = args.input_index
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}.")
        
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    
    codebook_path = args.input_codebook
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"Codebook file not found at {codebook_path}.")
    codebook_data = np.load(codebook_path)
    codebook_tokens = codebook_data['tokens']
    
    # 2. Compute velocities and trajectories matching the engine's MM logic
    print("Computing velocities and trajectories...")
    valid_mask = file_indices[:-1] == file_indices[1:]
    valid_mask = np.append(valid_mask, False)
    
    pose_vel = np.zeros_like(poses)
    trans_vel = np.zeros_like(trans)
    
    pose_vel[:-1][valid_mask[:-1]] = poses[1:][valid_mask[:-1]] - poses[:-1][valid_mask[:-1]]
    trans_vel[:-1][valid_mask[:-1]] = trans[1:][valid_mask[:-1]] - trans[:-1][valid_mask[:-1]]

    traj_15 = np.zeros_like(trans)
    traj_30 = np.zeros_like(trans)
    
    for offset, traj_array in [(15, traj_15), (30, traj_30)]:
        shifted_trans = np.roll(trans, -offset, axis=0)
        shifted_file_indices = np.roll(file_indices, -offset, axis=0)
        valid_traj_mask = (file_indices == shifted_file_indices)
        
        traj_array[valid_traj_mask] = shifted_trans[valid_traj_mask] - trans[valid_traj_mask]
        traj_array[~valid_traj_mask] = trans_vel[~valid_traj_mask] * offset

    # 3. Group frames by Region
    print("Grouping valid frames by Codebook Region...")
    region_to_frames = defaultdict(list)
    for idx, reg in enumerate(codebook_tokens):
        if reg != -1 and valid_mask[idx]:
            region_to_frames[reg].append(idx)
            
    regions = sorted(list(region_to_frames.keys()))
    print(f"Found {len(regions)} valid codebook regions.")

    # 4. Build Plausibility Graph
    graph = defaultdict(dict)
    
    # Fast sampling approach
    np.random.seed(42)
    max_source_samples = 200  # Number of sequences to forward-simulate per region
    
    # 4. Extract global feature vectors for fast L2 matrix operations
    # Combine the weights directly into the feature vectors so simple L2 distance works implicitly
    print("Building global continuous feature array...")
    # shape: (N, feature_dim)
    # MM Cost: pose(x1) + pose_vel(x1) + 10*trans_vel(x10) + 2*traj15(x2) + 2*traj30(x2)
    global_features = np.concatenate([
        poses, 
        pose_vel, 
        trans_vel * np.sqrt(10.0), 
        traj_15 * np.sqrt(2.0), 
        traj_30 * np.sqrt(2.0)
    ], axis=1).astype(np.float32)

    print("Building graph edges...")
    for reg_A in tqdm(regions, desc="Processing source regions"):
        frames_A = region_to_frames[reg_A]
        
        # Sample starting frames for Region A to avoid excessive computation
        num_start_samples = min(max_source_samples, len(frames_A))
        start_frames = np.random.choice(frames_A, num_start_samples, replace=False)
        
        # Collect window frames
        window_frames = []
        for i in start_frames:
            i_end = i + args.fast_forward_frames
            max_idx = i_end + args.transition_window_size
            if max_idx < len(file_indices) and file_indices[i] == file_indices[max_idx]:
                window_frames.extend(range(i_end, max_idx + 1))
                
        if not window_frames:
            continue
            
        window_frames = np.array(window_frames)
        feat_A_windows = global_features[window_frames]  # [W, Feature_Dim]
        
        for reg_B in regions:
            frames_B = region_to_frames[reg_B]
            if not frames_B:
                continue
                
            num_samples = min(args.num_candidates, len(frames_B))
            candidates_j = np.random.choice(frames_B, num_samples, replace=False)
            feat_B_candidates = global_features[candidates_j]  # [C, Feature_Dim]
            
            # Efficient pairwise squared euclidean distance calculation:
            # (a-b)^2 = a^2 + b^2 - 2ab
            # Using scipy.spatial.distance.cdist or einsum is incredibly fast here
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(feat_A_windows, feat_B_candidates, metric='sqeuclidean')
            
            best_cost = np.min(dist_matrix)
            if best_cost <= args.max_plausible_cost:
                graph[int(reg_A)][int(reg_B)] = float(best_cost)
                
    # 5. Save the Graph
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(dict(graph), f)
        
    num_edges = sum(len(targets) for targets in graph.values())
    print(f"Graph generation complete. Saved to {args.output_path}.")
    print(f"Total Regions (Nodes): {len(graph)}")
    print(f"Total Valid Transitions (Edges): {num_edges}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Graph Building: Map out codebook transition plausibilities.")
    parser.add_argument("--input_index", type=str, default=os.path.join("data", "index", "motion_index.npz"), help="Path to motion index")
    parser.add_argument("--input_codebook", type=str, default=os.path.join("data", "index", "codebook.npz"), help="Path to codebook region labels")
    parser.add_argument("--output_path", type=str, default=os.path.join("data", "index", "plausibility_graph.pkl"), help="Output path for the graph")
    
    # Engine simulation and transition pruning parameters
    parser.add_argument("--fast_forward_frames", type=int, default=30, help="Frames played before allowing a jump search (respects 30-frame lock)")
    parser.add_argument("--transition_window_size", type=int, default=15, help="Simulation transition search window")
    parser.add_argument("--num_candidates", type=int, default=100, help="Number of random candidate transition targets per region")
    parser.add_argument("--max_plausible_cost", type=float, default=6.0, help="Max standard MM cost before edge is considered an 'impossible' glitch transition")
    
    args = parser.parse_args()
    create_plausibilities(args)
