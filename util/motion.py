import numpy as np
import torch
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

def physics_contact_fix(poses, trans, body_model, verbose=True):
    """
    Standard physics post-processing to eliminate foot sliding and height drifting.
    Detects foot contacts using velocity, and interpolates a height correction
    across the sequence to keep the planted foot exactly over the ground (Y=0),
    while preserving the relative height of jumps.
    """
    if verbose: print("Running physics post-processing (Grounding/Foot-lock)...")
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
    in_contact = foot_vels < 0.02
    
    min_vel = np.min(foot_vels, axis=1)
    any_in_contact = min_vel < 0.02
    
    if np.sum(any_in_contact) == 0:
        return trans # Unlikely for a dance, but fallback

    contact_indices = np.where(any_in_contact)[0]
    corrections_y = np.zeros(num_frames)
    
    # Correction places the lowest point of the mesh on the floor (Y=0)
    for idx in contact_indices:
        corrections_y[idx] = -min_mesh_y[idx]
        
    # Interpolate corrections across jump frames smoothly
    interp_corrections_y = np.interp(np.arange(num_frames), contact_indices, corrections_y[contact_indices])
    
    # Low-pass filter the correction to avoid sudden snaps
    if len(interp_corrections_y) > 31:
        interp_corrections_y = savgol_filter(interp_corrections_y, 31, 3)
        
    fixed_trans = np.copy(trans)
    fixed_trans[:, 1] += interp_corrections_y

    # --- NEW: Horizontal Sliding Fix ---
    xz_corrections = np.zeros((num_frames, 2))
    contact_counts = np.zeros(num_frames)

    for j in range(4):
        contact_mask = in_contact[:, j]
        changes = np.diff(contact_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if contact_mask[0]:
            starts = np.insert(starts, 0, 0)
        if contact_mask[-1]:
            ends = np.append(ends, num_frames)
            
        for s, e in zip(starts, ends):
            ref_pos = foot_joints[s, j, [0, 2]]
            for f in range(s, e):
                curr_pos = foot_joints[f, j, [0, 2]]
                drift = curr_pos - ref_pos
                xz_corrections[f] -= drift
                contact_counts[f] += 1
                
    mask = contact_counts > 0
    xz_corrections[mask] = xz_corrections[mask] / contact_counts[mask][:, None]
    
    # Interpolate xz_corrections across non-contact frames
    if np.sum(mask) > 0:
        contact_indices_xz = np.where(mask)[0]
        interp_corrections_x = np.interp(np.arange(num_frames), contact_indices_xz, xz_corrections[contact_indices_xz, 0])
        interp_corrections_z = np.interp(np.arange(num_frames), contact_indices_xz, xz_corrections[contact_indices_xz, 1])
        
        if num_frames > 31:
            interp_corrections_x = savgol_filter(interp_corrections_x, 31, 3)
            interp_corrections_z = savgol_filter(interp_corrections_z, 31, 3)
            
        fixed_trans[:, 0] += interp_corrections_x
        fixed_trans[:, 2] += interp_corrections_z
    # -----------------------------------

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
