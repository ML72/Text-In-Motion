import argparse
import numpy as np
import os
import pickle
import time
import torch
import imageio
import pyrender
import smplx
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy
from scipy.signal import savgol_filter

def interpolate_pose(p1, p2, alpha):
    # Use proper spherical linear interpolation (SLERP) for joint rotations 
    # to prevent limbs from collapsing or contorting into inhuman shapes.
    r1 = R_scipy.from_rotvec(p1.reshape(-1, 3))
    r2 = R_scipy.from_rotvec(p2.reshape(-1, 3))
    
    # Compute relative rotation, scale its angle by alpha, and apply to r1
    r_rel = r1.inv() * r2
    r_interp = r1 * R_scipy.from_rotvec(r_rel.as_rotvec() * alpha)
    return r_interp.as_rotvec().flatten()

def interpolate_trans_vel(v1, v2, alpha):
    return (1 - alpha) * v1 + alpha * v2

def generate_motion(num_frames, run_name):
    print("Loading indexed data...")
    index_path = os.path.join("data", "index", "motion_index.npz")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}. Please run create_index.py first.")
        
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    
    print("Computing velocities...")
    # Check boundaries so we don't compute velocity across different motion files
    valid_mask = file_indices[:-1] == file_indices[1:]
    # Append False for the last element because it has no 'next' file to compare internally
    valid_mask = np.append(valid_mask, False)
    
    pose_vel = np.zeros_like(poses)
    trans_vel = np.zeros_like(trans)
    
    pose_vel[:-1][valid_mask[:-1]] = poses[1:][valid_mask[:-1]] - poses[:-1][valid_mask[:-1]]
    trans_vel[:-1][valid_mask[:-1]] = trans[1:][valid_mask[:-1]] - trans[:-1][valid_mask[:-1]]
    
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
    
    search_interval = 5 # Start searching more frequently for unique dances
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
            num_interp = 2
            for i in range(1, num_interp + 1):
                alpha = i / (num_interp + 1)
                interp_p = interpolate_pose(gen_poses[-1], next_pose, alpha)
                interp_v = interpolate_trans_vel(trans_vel[curr_idx], trans_vel[chosen_idx], alpha)
                
                current_absolute_trans += interp_v
                gen_poses.append(interp_p)
                gen_trans.append(current_absolute_trans.copy())
                
                frames_generated += 1
                pbar.update(1)
                if frames_generated >= num_frames:
                    break
                    
        if frames_generated >= num_frames:
            break
            
        current_absolute_trans += next_trans_vel
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
        gen_poses_np = savgol_filter(gen_poses_np, window_length, 3, axis=0) # polyorder 3 preserves hits
        gen_trans_np = savgol_filter(gen_trans_np, window_length, 3, axis=0)
        
    gen_poses = gen_poses_np.tolist()
    gen_trans = gen_trans_np.tolist()

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
    print("Rendering video...")
    try:
        body_model = smplx.create("models", model_type='smpl', gender='neutral', batch_size=1)
    except Exception as e:
        print("Visualization skipped. Could not load SMPL models.")
        print(f"Error: {e}")
        return

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    rotation = tra.rotation_matrix(np.radians(-20 * np.pi / 180), [1, 0, 0])
    camera_pose = np.eye(4) @ rotation
    camera_pose[1, 3] = 1.5
    camera_pose[2, 3] = 3.0
    cam_node = scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    writer = imageio.get_writer(mp4_out, fps=60)
    
    for i in tqdm(range(len(gen_poses)), desc="Rendering frames"):
        pose = torch.tensor(gen_poses[i:i+1], dtype=torch.float32)
        trans = torch.tensor(gen_trans[i:i+1], dtype=torch.float32)

        global_orient = pose[:, :3]
        body_pose = pose[:, 3:]

        output = body_model(global_orient=global_orient,
                            body_pose=body_pose,
                            transl=trans)

        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = body_model.faces

        mesh = trimesh.Trimesh(vertices, faces)
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        node = scene.add(mesh_node)
        color, _ = renderer.render(scene)
        writer.append_data(color)
        scene.remove_node(node)

    writer.close()
    renderer.delete()
    print(f"Saved visualization to {mp4_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classical motion-matching dances")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to generate")
    
    default_name = f"sample_{time.strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--run_name", type=str, default=default_name, help="Run name suffix")
    
    args = parser.parse_args()
    
    generate_motion(args.num_frames, args.run_name)