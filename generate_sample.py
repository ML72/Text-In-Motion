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

def physics_contact_fix(poses, trans, body_model):
    """
    Standard physics post-processing to eliminate foot sliding and height drifting.
    Detects foot contacts using velocity, and interpolates a height correction
    across the sequence to keep the planted foot exactly over the ground (Y=0),
    while preserving the relative height of jumps.
    """
    print("Running physics post-processing (Grounding/Foot-lock)...")
    poses_tensor = torch.tensor(poses, dtype=torch.float32)
    trans_tensor = torch.tensor(trans, dtype=torch.float32)
    num_frames = poses_tensor.shape[0]

    # Process in chunks if memory is limited, but 1000 frames is small enough for SMPL
    output = body_model(
        global_orient=poses_tensor[:, :3],
        body_pose=poses_tensor[:, 3:],
        transl=trans_tensor
    )
    
    # Get joints and vertices
    joints = output.joints.detach().numpy()
    vertices = output.vertices.detach().numpy()
    
    # joint indices: 7=L_Ankle, 8=R_Ankle, 10=L_Foot, 11=R_Foot
    foot_joints = joints[:, [7, 8, 10, 11], :]
    
    # Calculate velocities
    foot_vels = np.linalg.norm(np.diff(foot_joints, axis=0), axis=-1)
    foot_vels = np.vstack([foot_vels[0:1], foot_vels]) # pad first frame
    
    # Find the lowest vertex at each frame for absolute ground collision
    min_mesh_y = np.min(vertices[:, :, 1], axis=1)
    
    # Contact heuristic: If 1) any foot velocity is near 0 AND 2) it is near the bottom
    min_vel = np.min(foot_vels, axis=1)
    in_contact = min_vel < 0.02
    
    if np.sum(in_contact) == 0:
        return trans # Unlikely for a dance, but fallback

    contact_indices = np.where(in_contact)[0]
    corrections = np.zeros(num_frames)
    
    # Correction places the lowest point of the mesh on the floor (Y=0)
    # The lowest point is usually the sole of the planted foot or the toe.
    for idx in contact_indices:
        corrections[idx] = -min_mesh_y[idx]
        
    # Interpolate corrections across jump frames smoothly
    interp_corrections = np.interp(np.arange(num_frames), contact_indices, corrections[contact_indices])
    
    # Low-pass filter the correction to avoid sudden snaps
    if len(interp_corrections) > 31:
        interp_corrections = savgol_filter(interp_corrections, 31, 3)
        
    fixed_trans = np.copy(trans)
    fixed_trans[:, 1] += interp_corrections
    return fixed_trans.tolist()

def smooth_poses_quaternion(poses_np, window_length, polyorder):
    """
    Applies Savitzky-Golay filter on quaternion representations of the poses
    instead of raw rotation vectors. Raw rotation vectors suffer from Gimbal Lock
    and 2*pi discontinuities (where interpolating from 3.14 to -3.14 passes
    through 0 instead of taking the shortest path), which severely contorts the mesh.
    """
    num_frames = poses_np.shape[0]
    num_joints = poses_np.shape[1] // 3
    
    # Convert rotation vectors to quaternions: (N, 24, 4)
    r = R_scipy.from_rotvec(poses_np.reshape(-1, 3))
    quats = r.as_quat().reshape(num_frames, num_joints, 4)
    
    # Align hemispheres: q and -q represent the same rotation. 
    # If the quaternion flips signs between frames, the linear filter will interpolate through 0.
    for i in range(1, num_frames):
        dot_products = np.sum(quats[i] * quats[i-1], axis=-1, keepdims=True)
        quats[i] = np.where(dot_products < 0, -quats[i], quats[i])
        
    # Safely filter the continuous quaternion representations
    quats_filtered = savgol_filter(quats, window_length, polyorder, axis=0)
    
    # Re-normalize quaternions to ensure they remain valid rotations
    quats_filtered /= np.linalg.norm(quats_filtered, axis=-1, keepdims=True)
    
    # Convert back to rotation vectors for SMPL
    r_filtered = R_scipy.from_quat(quats_filtered.reshape(-1, 4))
    return r_filtered.as_rotvec().reshape(num_frames, poses_np.shape[1])

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
            num_interp = 8 # Increase transition window for smooth inertial blending
            for i in range(1, num_interp + 1):
                alpha = i / (num_interp + 1)
                # Apply an ease-in-out spherical decay to make the transition less linear
                smooth_alpha = alpha * alpha * (3 - 2 * alpha)
                
                interp_p = interpolate_pose(gen_poses[-1], next_pose, smooth_alpha)
                interp_v = interpolate_trans_vel(trans_vel[curr_idx], trans_vel[chosen_idx], smooth_alpha)
                
                # Accumulate X/Z, but directly interpolate the absolute Y (height)
                # to prevent floating/sinking over time across jumps.
                current_absolute_trans[0] += interp_v[0]
                current_absolute_trans[2] += interp_v[2]
                current_absolute_trans[1] = (1 - alpha) * trans[curr_idx][1] + alpha * trans[chosen_idx][1]
                
                gen_poses.append(interp_p)
                gen_trans.append(current_absolute_trans.copy())
                
                frames_generated += 1
                pbar.update(1)
                if frames_generated >= num_frames:
                    break
                    
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
    camera_pose[1, 3] = 1.0
    camera_pose[2, 3] = 3.0
    cam_node = scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=camera_pose)

    # Add ground guide lines for better motion perception
    try:
        import trimesh.creation
        sm_cyls = []
        for x in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.01, height=20.0)
            # Cylinder is along Z by default, just translate along X
            cyl.apply_translation([x, 0, 0])
            sm_cyls.append(cyl)
        for z in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.01, height=20.0)
            # Rotate 90 degrees around Y to point along X, then translate along Z
            cyl.apply_transform(tra.rotation_matrix(np.pi/2, [0, 1, 0]))
            cyl.apply_translation([0, 0, z])
            sm_cyls.append(cyl)
        grid_mesh = trimesh.util.concatenate(sm_cyls)
        
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0], 
            metallicFactor=0.0, 
            roughnessFactor=1.0)
        grid_mesh_node = pyrender.Mesh.from_trimesh(grid_mesh, material=material, smooth=False)
        scene.add(grid_mesh_node)
    except Exception as e:
        print(f"Skipping grid generation: {e}")

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