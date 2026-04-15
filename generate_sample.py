import argparse
import numpy as np
import os
import pickle
import time
import smplx
from scipy.signal import savgol_filter
from tqdm import tqdm
from util.motion import interpolate_pose, interpolate_trans_vel, physics_contact_fix, smooth_poses_quaternion

def generate_motion(num_frames, run_name):
    print("Loading indexed data...")
    index_path = os.path.join("data", "index", "motion_index.npz")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}. Please run create_index.py first.")
        
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    
    print("Computing velocities and trajectories...")
    # Check boundaries so we don't compute velocity across different motion files
    valid_mask = file_indices[:-1] == file_indices[1:]
    # Append False for the last element because it has no 'next' file to compare internally
    valid_mask = np.append(valid_mask, False)
    
    pose_vel = np.zeros_like(poses)
    trans_vel = np.zeros_like(trans)
    
    pose_vel[:-1][valid_mask[:-1]] = poses[1:][valid_mask[:-1]] - poses[:-1][valid_mask[:-1]]
    trans_vel[:-1][valid_mask[:-1]] = trans[1:][valid_mask[:-1]] - trans[:-1][valid_mask[:-1]]

    # Motion Matching Trajectories: compute future positional offsets (15 and 30 frames ahead)
    # This prevents the algorithm from picking a frame that visually matches NOW, 
    # but abruptly shoots off in the wrong X/Z direction 10 frames later.
    traj_15 = np.zeros_like(trans)
    traj_30 = np.zeros_like(trans)
    
    for offset, traj_array in [(15, traj_15), (30, traj_30)]:
        shifted_trans = np.roll(trans, -offset, axis=0)
        shifted_file_indices = np.roll(file_indices, -offset, axis=0)
        valid_traj_mask = (file_indices == shifted_file_indices)
        
        traj_array[valid_traj_mask] = shifted_trans[valid_traj_mask] - trans[valid_traj_mask]
        # If we're too close to a boundary, approximate the future by scaling the current velocity
        traj_array[~valid_traj_mask] = trans_vel[~valid_traj_mask] * offset
    
    gen_poses = []
    gen_trans = []
    
    # 1. Initialize random starting frame
    curr_idx = np.random.randint(0, len(poses) - 1)
    while not valid_mask[curr_idx]:
        curr_idx = np.random.randint(0, len(poses) - 1)
        
    gen_poses.append(poses[curr_idx])
    current_absolute_trans = trans[curr_idx].copy()  # Uses correct initial root height!
    gen_trans.append(current_absolute_trans.copy())
    
    top_k = 10
    pose_threshold = 4.0 # L2 distance heuristic for insertion gap

    pbar = tqdm(total=num_frames, desc="Generating motion sequence")
    frames_generated = 1
    
    search_interval = 30 # Increased to stitch longer sequences together
    frames_since_last_jump = 0

    while frames_generated < num_frames:
        target_idx = curr_idx + 1
        if not valid_mask[curr_idx]:
            # If we somehow hit the end of a clip, jump arbitrarily
            target_idx = np.random.randint(0, len(poses) - 1)
            while not valid_mask[target_idx]:
                target_idx = np.random.randint(0, len(poses) - 1)
            chosen_idx = target_idx
            frames_since_last_jump = 0
            
        elif frames_since_last_jump >= search_interval:
            target_pose = poses[target_idx]
            target_vel = pose_vel[target_idx]
            
            # 2. Motion matching criterion: minimize pose + velocity difference
            dist = np.mean((poses - target_pose)**2, axis=1) + np.mean((pose_vel - target_vel)**2, axis=1)
            dist += np.mean((trans_vel - trans_vel[target_idx])**2, axis=1) * 10.0 # Match translational velocity to prevent sliding
            
            # Trajectory matching penalty: evaluate future momentum vectors to prevent X/Z darting
            dist += np.mean((traj_15 - traj_15[target_idx])**2, axis=1) * 2.0
            dist += np.mean((traj_30 - traj_30[target_idx])**2, axis=1) * 2.0
            
            dist[~valid_mask] = np.inf # Exclude corrupted boundary transitions
            
            # Penalize loops by avoiding recent moving average
            recent_window = np.array(gen_poses[-20:])
            moving_avg = np.mean(recent_window, axis=0) # (72,)
            # Exponentiate the moving average distance so the penalty decays gracefully 
            # and doesn't completely break the primary motion matching objective.
            avg_dist = np.mean((poses - moving_avg)**2, axis=1)
            dist += 0.2 * np.exp(-5.0 * avg_dist) # Penalty multiplier
            
            # Hysteresis (Switching Cost): heavily subsidize the natural next frame 
            # to stick to the current clip unless a significantly better match is found.
            dist[target_idx] -= 5.0
            
            # Nucleus / top-k sampling for variety
            best_indices = np.argpartition(dist, top_k)[:top_k]
            chosen_idx = np.random.choice(best_indices)
            
            if chosen_idx != target_idx:
                frames_since_last_jump = 0
            else:
                frames_since_last_jump += 1
        else:
            # Native clip continuation (avoids "hunting" jitter)
            chosen_idx = target_idx
            frames_since_last_jump += 1
            
        next_pose = poses[chosen_idx]
        next_trans_vel = trans_vel[chosen_idx]
        
        # 3. Check for discontinuity to interpolate frames
        # Even among top-k, if the gap is sudden, dynamically interpolate
        pose_dist = np.linalg.norm(gen_poses[-1] - next_pose)
        
        if pose_dist > pose_threshold and chosen_idx != target_idx:
            num_interp = 10 # Increase transition window for smooth inertial blending
            
            # Industry Standard: Inertial Blending
            # Instead of standard crossfading which blurs movement, we immediately snap to the 
            # new animation clip and advance it, while applying a mathematically decaying 
            # offset from the old posture. This preserves the absolute velocities and high-frequency 
            # details (like foot plants) of the new clip!
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
                    # Alpha decays from 1.0 (source) down to 0.0 (target)
                    alpha = 1.0 - (i / (num_interp + 1))
                    
                    # Fast Quintic ease-out decay to quickly restore the target animation's gait
                    decay = alpha ** 3 
                    
                    new_p = poses[chosen_idx + i]
                    # Interpolate from target back towards source by the decay amount
                    interp_p = interpolate_pose(new_p, source_pose, decay)
                    
                    new_v = trans_vel[chosen_idx + i]
                    interp_v = new_v + (source_vel - new_v) * decay
                    
                    # Accumulate X/Z, but directly interpolate the absolute Y (height)
                    current_absolute_trans[0] += interp_v[0]
                    current_absolute_trans[2] += interp_v[2]
                    
                    target_y = trans[chosen_idx + i][1]
                    current_absolute_trans[1] = target_y + (source_trans_y - target_y) * decay
                    
                    gen_poses.append(interp_p)
                    gen_trans.append(current_absolute_trans.copy())
                    
                    frames_generated += 1
                    pbar.update(1)
                    if frames_generated >= num_frames:
                        break
                        
                curr_idx = chosen_idx + num_interp
                if frames_generated >= num_frames:
                    break
                continue
                    
        if frames_generated >= num_frames:
            break
            
        current_absolute_trans[0] += next_trans_vel[0]
        current_absolute_trans[2] += next_trans_vel[2]
        current_absolute_trans[1] = trans[chosen_idx][1]
        
        gen_poses.append(next_pose)
        gen_trans.append(current_absolute_trans.copy())
        
        curr_idx = chosen_idx
        frames_generated += 1
        pbar.update(1)
        
    pbar.close()

    # 4. Post-processing: Savitzky-Golay filter to smooth out macro jitter while preserving peaks
    print("Applying Savitzky-Golay filter to smooth transitions...")
    gen_poses_np = np.array(gen_poses)
    gen_trans_np = np.array(gen_trans)
    
    window_length = 15 # Must be odd
    if len(gen_poses_np) >= window_length:
        gen_poses_np = smooth_poses_quaternion(gen_poses_np, window_length, 3)
        gen_trans_np = savgol_filter(gen_trans_np, window_length, 3, axis=0)
        
    gen_poses = gen_poses_np.tolist()
    gen_trans = gen_trans_np.tolist()

    # 5. Physics check / Contact constraint resolution
    try:
        body_model = smplx.create("models", model_type='smpl', gender='neutral', batch_size=num_frames)
        gen_trans = physics_contact_fix(gen_poses, gen_trans, body_model)
    except Exception as e:
        print(f"Skipping physics post-processing: Could not apply foot contacts. Ground truth SMPL models may be missing or incompatible batch size. Error: {e}")

    # Create export directory
    os.makedirs("results", exist_ok=True)
    pkl_out = os.path.join("results", f"{run_name}.pkl")
    mp4_out = os.path.join("results", f"{run_name}.mp4")
    
    # Save standard AIST++ dictionary
    output_dict = {
        'smpl_poses': np.array(gen_poses),
        'smpl_trans': np.array(gen_trans),
        'smpl_scaling': np.array([1.0])
    }
    
    with open(pkl_out, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"Saved motion data to {pkl_out}")

    # Visualizer
    from util.render import render_animation
    render_animation(gen_poses, gen_trans, mp4_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classical motion-matching dances")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to generate")
    
    default_name = f"sample_{time.strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--run_name", type=str, default=default_name, help="Run name suffix")
    
    args = parser.parse_args()
    
    generate_motion(args.num_frames, args.run_name)