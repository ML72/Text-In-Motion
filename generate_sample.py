import argparse
import numpy as np
import os
import pickle
import time
import json
import smplx
import heapq
from scipy.signal import savgol_filter
from tqdm import tqdm
from util.motion import interpolate_pose, interpolate_trans_vel, physics_contact_fix, smooth_poses_quaternion
from util.codebook import fill_invalid_regions, compute_dna

def dijkstra_shortest_path(graph, start, target):
    queue = [(0.0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == target:
            return path
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return []

def generate_motion(num_frames, run_name, dna_string=None, input_text=None, gender='neutral'):
    print("Loading indexed data...")
    index_path = os.path.join("data", "index", "motion_index.npz")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}. Please run create_index.py first.")
        
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    frame_indices = index_data['frame_indices']
    
    codebook_path = os.path.join("data", "index", "codebook.npz")
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"Codebook file not found at {codebook_path}. Please run create_codebook.py first.")
    codebook_data = np.load(codebook_path)
    codebook_tokens = codebook_data['tokens']
    
    file_names_path = os.path.join("data", "index", "file_names.json")
    with open(file_names_path, 'r') as f:
        file_names = json.load(f)
        
    graph_path = os.path.join("data", "index", "plausibility_graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Plausibility graph not found at {graph_path}. Run create_plausibilities.py first.")
    with open(graph_path, 'rb') as f:
        plausibility_graph = pickle.load(f)
        
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

    MAX_PLAUSIBLE_COST = 4.0
    
    # Precompute region reference poses for graceful fallbacks
    print("Computing region reference poses...")
    region_centers = {}
    for r in np.unique(codebook_tokens):
        if r != -1:
            region_mask = (codebook_tokens == r) & valid_mask
            if np.any(region_mask):
                region_centers[r] = np.mean(poses[region_mask], axis=0)

    gen_poses = []
    gen_trans = []
    gen_metadata = []
    
    executed_dna = []
    frames_since_jump = 0

    mode = "A"
    dna_queue = []
    if input_text is not None:
        mode = "C"
        input_text = input_text.strip()
        dna_queue = list(input_text.encode('utf-8'))
        print(f"Mode C: Text Guided Generation. Target DNA: {dna_queue}")
    elif dna_string:
        mode = "B"
        dna_queue = [int(x.strip()) for x in dna_string.split(',') if x.strip()]
        print(f"Mode B: Guided Generation. Target DNA: {dna_queue}")
    else:
        print("Mode A: Autonomous Exploration.")

    def compute_mm_dist(target_idx):
        dist = np.mean((poses - poses[target_idx])**2, axis=1) + np.mean((pose_vel - pose_vel[target_idx])**2, axis=1)
        dist += np.mean((trans_vel - trans_vel[target_idx])**2, axis=1) * 10.0
        dist += np.mean((traj_15 - traj_15[target_idx])**2, axis=1) * 2.0
        dist += np.mean((traj_30 - traj_30[target_idx])**2, axis=1) * 2.0
        return dist

    if mode == "A":
        curr_idx = np.random.randint(0, len(poses) - 1)
        while not valid_mask[curr_idx] or codebook_tokens[curr_idx] == -1:
            curr_idx = np.random.randint(0, len(poses) - 1)
        
        current_region = int(codebook_tokens[curr_idx])
        executed_dna.append(current_region)
    else:
        if not dna_queue:
            raise ValueError("DNA queue cannot be empty in Mode B.")
        T_start = dna_queue.pop(0)
        
        valid_starts = np.where((codebook_tokens == T_start) & valid_mask)[0]
        if len(valid_starts) == 0:
            raise ValueError(f"No valid starting frames found for region {T_start}")
        curr_idx = np.random.choice(valid_starts)
        
        current_region = T_start
        executed_dna.append(current_region)

    gen_poses.append(poses[curr_idx])
    current_absolute_trans = trans[curr_idx].copy()
    gen_trans.append(current_absolute_trans.copy())
    gen_metadata.append({
        "engine_frame": 0,
        "source_motion_id": file_names[int(file_indices[curr_idx])],
        "source_frame_idx": int(frame_indices[curr_idx]),
        "codebook_region": int(codebook_tokens[curr_idx])
    })
    
    frames_generated = 1

    pbar = tqdm(total=num_frames if mode == "A" else None, desc="Generating motion sequence")
    
    pose_threshold = 4.0

    while True:
        if mode == "A" and frames_generated >= num_frames:
            break
        if mode in ["B", "C"] and len(dna_queue) == 0:
            break
            
        target_idx = curr_idx + 1
        forced_jump = False
        if not valid_mask[curr_idx] or codebook_tokens[target_idx] == -1:
            forced_jump = True
            
        frames_since_jump += 1
        chosen_idx = target_idx
        jumped = False
        
        if mode == "A":
            if frames_since_jump >= 30 or forced_jump:
                dist = compute_mm_dist(target_idx) if not forced_jump else compute_mm_dist(curr_idx)
                
                # Apply novelty penalty to avoid repetitive loops
                novelty_penalty = np.zeros_like(dist)
                for i, r in enumerate(reversed(executed_dna[-20:])):
                    novelty_penalty[codebook_tokens == r] += 5.0 / (i + 1)
                dist += novelty_penalty
                
                dist[~valid_mask] = np.inf
                dist[codebook_tokens == -1] = np.inf
                dist[codebook_tokens == current_region] = np.inf
                
                sort_idx = np.argsort(dist)
                found_good = False
                for i in range(min(10, len(sort_idx))):
                    candidate_idx = sort_idx[i]
                    candidate_cost = dist[candidate_idx]
                    
                    if candidate_cost == np.inf:
                        break
                        
                    if (forced_jump and candidate_cost != np.inf and i == 0) or candidate_cost < MAX_PLAUSIBLE_COST:
                        cand_pose = poses[candidate_idx]
                        pose_dist = np.linalg.norm(gen_poses[-1] - cand_pose)
                        
                        if pose_dist <= pose_threshold or (forced_jump and i == 0):
                            chosen_idx = candidate_idx
                            current_region = int(codebook_tokens[chosen_idx])
                            executed_dna.append(current_region)
                            frames_since_jump = 0
                            jumped = True
                            found_good = True
                            break
        elif mode in ["B", "C"]:
            T_next = dna_queue[0]
            
            if 30 <= frames_since_jump <= 60 or forced_jump:
                dist = compute_mm_dist(target_idx) if not forced_jump else compute_mm_dist(curr_idx)
                
                dist[~valid_mask] = np.inf
                dist[codebook_tokens != T_next] = np.inf
                
                sort_idx = np.argsort(dist)
                found_good = False
                for i in range(min(10, len(sort_idx))):
                    candidate_idx = sort_idx[i]
                    candidate_cost = dist[candidate_idx]
                    
                    if candidate_cost == np.inf:
                        break
                        
                    if (forced_jump and candidate_cost != np.inf and i == 0) or candidate_cost < MAX_PLAUSIBLE_COST:
                        cand_pose = poses[candidate_idx]
                        pose_dist = np.linalg.norm(gen_poses[-1] - cand_pose)
                        
                        if pose_dist <= pose_threshold or (forced_jump and i == 0):
                            chosen_idx = candidate_idx
                            current_region = int(codebook_tokens[chosen_idx])
                            executed_dna.append(T_next)
                            dna_queue.pop(0)
                            frames_since_jump = 0
                            jumped = True
                            found_good = True
                            break
                            
                if not found_good:
                    forced_jump = True

            if not jumped and (frames_since_jump >= 60 or forced_jump):
                dist = compute_mm_dist(target_idx) if not forced_jump else compute_mm_dist(curr_idx)
                
                dist[~valid_mask] = np.inf
                dist[codebook_tokens == -1] = np.inf
                dist[codebook_tokens == current_region] = np.inf
                
                best_idx = np.argmin(dist)
                min_cost = dist[best_idx]
                
                chosen_idx = best_idx
                new_region = int(codebook_tokens[chosen_idx])
                executed_dna.append(new_region)
                current_region = new_region
                
                path = dijkstra_shortest_path(plausibility_graph, current_region, T_next)
                
                # Graceful Fallback: Closest Neighbor Replacement
                if not path and current_region != T_next:
                    print(f"\nWarning: Target region {T_next} is unreachable. Searching for closest proxy...")
                    best_proxy = None
                    best_proxy_dist = np.inf
                    
                    if T_next in region_centers:
                        t_next_center = region_centers[T_next]
                        valid_nodes = list(plausibility_graph.keys())
                        
                        # Find the mathematically most similar region that is actually accessible from here
                        for proxy in valid_nodes:
                            if proxy != current_region and proxy in region_centers:
                                dist_to_target = np.linalg.norm(region_centers[proxy] - t_next_center)
                                
                                if dist_to_target < best_proxy_dist:
                                    # Verify it's reachable before committing
                                    proxy_path = dijkstra_shortest_path(plausibility_graph, current_region, proxy)
                                    if proxy_path:
                                        best_proxy_dist = dist_to_target
                                        best_proxy = proxy
                                        path = proxy_path
                                        
                    if best_proxy is not None:
                        print(f"-> Substituted {T_next} with {best_proxy} (Distance: {best_proxy_dist:.2f})")
                        T_next = best_proxy
                        dna_queue[0] = best_proxy  # Overwrite the unreachable token so the engine consumes the proxy instead
                    else:
                        print(f"-> No valid proxy found. Skipping DNA token {T_next} entirely.")
                        dna_queue.pop(0)
                        path = []

                if path is None:
                    path = []
                
                dna_queue = path + dna_queue
                frames_since_jump = 0
                jumped = True

        if not jumped:
            current_region = int(codebook_tokens[chosen_idx])

        next_pose = poses[chosen_idx]
        next_trans_vel = trans_vel[chosen_idx]
        
        pose_dist = np.linalg.norm(gen_poses[-1] - next_pose)
        
        if pose_dist > pose_threshold and chosen_idx != target_idx:
            num_interp = 20  # Increased from 10: more frames of interpolation for coherence
            can_interp = True
            for i in range(1, num_interp + 1):
                if not (chosen_idx + i < len(poses) and valid_mask[chosen_idx + i - 1]):
                    can_interp = False
                    break
            
            if can_interp:
                source_pose = gen_poses[-1]
                source_trans_y = current_absolute_trans[1]
                source_vel = trans_vel[curr_idx].copy()
                
                for i in range(1, num_interp + 1):
                    # Use industry standard Ken Perlin's Smootherstep for C2 continuous ease-in/ease-out
                    t = i / (num_interp + 1)
                    t_smooth = t * t * t * (t * (t * 6 - 15) + 10)
                    decay = 1.0 - t_smooth
                    
                    new_p = poses[chosen_idx + i]
                    interp_p = interpolate_pose(new_p, source_pose, decay)
                    
                    new_v = trans_vel[chosen_idx + i]
                    interp_v = new_v + (source_vel - new_v) * decay
                    
                    current_absolute_trans[0] += interp_v[0]
                    current_absolute_trans[2] += interp_v[2]
                    
                    target_y = trans[chosen_idx + i][1]
                    current_absolute_trans[1] = target_y + (source_trans_y - target_y) * decay
                    
                    gen_poses.append(interp_p)
                    gen_trans.append(current_absolute_trans.copy())
                    gen_metadata.append({
                        "engine_frame": frames_generated,
                        "source_motion_id": file_names[int(file_indices[chosen_idx + i])],
                        "source_frame_idx": int(frame_indices[chosen_idx + i]),
                        "codebook_region": int(codebook_tokens[chosen_idx + i])
                    })
                    
                    frames_generated += 1
                    pbar.update(1)
                    if mode == "A" and frames_generated >= num_frames:
                        break
                        
                curr_idx = chosen_idx + num_interp
                if mode == "A" and frames_generated >= num_frames:
                    break
                continue
                    
        if mode == "A" and frames_generated >= num_frames:
            break
            
        current_absolute_trans[0] += next_trans_vel[0]
        current_absolute_trans[2] += next_trans_vel[2]
        current_absolute_trans[1] = trans[chosen_idx][1]
        
        gen_poses.append(next_pose)
        gen_trans.append(current_absolute_trans.copy())
        gen_metadata.append({
            "engine_frame": frames_generated,
            "source_motion_id": file_names[int(file_indices[chosen_idx])],
            "source_frame_idx": int(frame_indices[chosen_idx]),
            "codebook_region": int(codebook_tokens[chosen_idx])
        })
        
        curr_idx = chosen_idx
        frames_generated += 1
        pbar.update(1)
        
    pbar.close()

    print("Applying Savitzky-Golay filter to smooth transitions...")
    gen_poses_np = np.array(gen_poses)
    gen_trans_np = np.array(gen_trans)
    
    window_length = 31  # Increased from 15: larger moving window for silky smooth results
    if len(gen_poses_np) >= window_length:
        gen_poses_np = smooth_poses_quaternion(gen_poses_np, window_length, 3)
        gen_trans_np = savgol_filter(gen_trans_np, window_length, 3, axis=0)
        
    # Translate entire sequence so the dancer always starts at the origin (0, y, 0)
    # This prevents the camera from getting too close or too far away depending on dataset coordinate
    gen_trans_np[:, 0] -= gen_trans_np[0, 0]
    gen_trans_np[:, 2] -= gen_trans_np[0, 2]

    gen_poses = gen_poses_np.tolist()
    gen_trans = gen_trans_np.tolist()

    try:
        body_model = smplx.create("models", model_type='smpl', gender='neutral', batch_size=frames_generated)
        gen_trans = physics_contact_fix(gen_poses, gen_trans, body_model)
    except Exception as e:
        print(f"Skipping physics post-processing: {e}")

    results_dir = os.path.join("results", "samples", run_name)
    os.makedirs(results_dir, exist_ok=True)
    pkl_out = os.path.join(results_dir, "dance_moves.pkl")
    mp4_out = os.path.join(results_dir, "dance_visualization.mp4")
    json_out = os.path.join(results_dir, "run_data.json")

    if mode == "A":
        logged_dna = None
        logged_text = None
    elif mode == "B":
        logged_dna = ",".join(str(x.strip()) for x in dna_string.split(',') if x.strip())
        logged_text = None
    else:
        logged_dna = ",".join(str(x) for x in input_text.encode('utf-8'))
        logged_text = input_text

    run_data = {
        "input_text": logged_text,
        "input_dna": logged_dna,
        "executed_dna": ",".join(str(x) for x in executed_dna),
        "frame_data": gen_metadata
    }
    with open(json_out, 'w') as f:
        json.dump(run_data, f, indent=4)
    print(f"Saved run data to {json_out}")

    output_dict = {
        'smpl_poses': np.array(gen_poses),
        'smpl_trans': np.array(gen_trans),
        'smpl_scaling': np.array([1.0])
    }
    
    with open(pkl_out, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"Saved motion data to {pkl_out}")

    from util.render import render_animation
    render_animation(gen_poses, gen_trans, mp4_out, gender=gender)

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Generate classical motion-matching dances")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to generate (Mode A only)")
    
    default_name = f"sample_{time.strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--run_name", type=str, default=default_name, help="Run name suffix")
    parser.add_argument("--input_dna", type=str, default=None, help="Comma separated Codebook Regions (Mode B only)")
    parser.add_argument("--input_text", type=str, default=None, help="String to guide generation (Mode C only)")
    parser.add_argument("--gender", type=str, choices=["neutral", "male", "female"], default="neutral", help="Gender of the SMPL model to use (neutral, male, female)")
    
    args = parser.parse_args()
    
    provided_args = sum([
        '--num_frames' in sys.argv,
        args.input_dna is not None,
        args.input_text is not None
    ])
    
    if provided_args > 1:
        parser.error("Cannot specify more than one of --num_frames, --input_dna, or --input_text simultaneously. These options are mutually exclusive (Modes A, B, and C).")
        
    generate_motion(args.num_frames, args.run_name, args.input_dna, args.input_text, args.gender)
