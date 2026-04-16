import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import pyrender
import smplx
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm
from collections import Counter

def render_animation_custom(gen_poses, gen_trans, mp4_out):
    try:
        body_model = smplx.create("models", model_type='smpl', gender='neutral', batch_size=1)
    except Exception as e:
        print("Visualization skipped. Could not load SMPL models.")
        print(f"Error: {e}")
        return

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    # Adjusting camera to prevent the human from floating/cutting off
    # Looking down slightly more, moved up a bit.
    rotation = tra.rotation_matrix(np.radians(-25 * np.pi / 180), [1, 0, 0])
    camera_pose = np.eye(4) @ rotation
    camera_pose[1, 3] = 1.5  # Moved camera up
    camera_pose[2, 3] = 3.5  # Moved camera out slightly
    cam_node = scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=camera_pose)

    # Grid lines
    try:
        import trimesh.creation
        sm_cyls = []
        for x in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.01, height=20.0)
            cyl.apply_translation([x, 0, 0])
            sm_cyls.append(cyl)
        for z in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.01, height=20.0)
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
    
    base_x, base_z = None, None
    for i in range(len(gen_poses)):
        pose = torch.tensor(gen_poses[i:i+1], dtype=torch.float32)
        trans = torch.tensor(gen_trans[i:i+1], dtype=torch.float32)
        
        # User requested bringing the person down if we didn't fix camera angle fully
        # Add a vertical fix and center the starting position at the origin
        if base_x is None:
            base_x = trans[0, 0].item()
            base_z = trans[0, 2].item()
        
        trans[0, 0] -= base_x
        trans[0, 2] -= base_z
        trans[0, 1] -= 0.6  # Adjust vertical translation to make sure feet touch the ground

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

def main():
    parser = argparse.ArgumentParser(description="Debug and visualize Codebook regions.")
    parser.add_argument("--num-regions", type=int, default=5, help="Number of codebook regions to sample")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of random frames to sample without replacement")
    parser.add_argument("--window", type=int, default=20, help="Length of visualization window for each frame")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic execution")
    args = parser.parse_args()

    # Determinism
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    index_path = os.path.join("data", "index", "motion_index.npz")
    codebook_path = os.path.join("data", "index", "codebook.npz")
    out_dir = os.path.join("results", "codebook")

    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}. Please run create_index.py first.")
        return
    if not os.path.exists(codebook_path):
        print(f"Error: Codebook file not found at {codebook_path}. Please run create_codebook.py first.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    
    codebook_data = np.load(codebook_path)
    tokens = codebook_data['tokens']
    
    # Analyze region distribution
    print("Analyzing codebook distribution...")
    valid_tokens = tokens[tokens != -1]
    
    if len(valid_tokens) == 0:
        print("Error: No valid tokens found in the codebook.")
        return
        
    counts = Counter(valid_tokens)
    unique_regions = sorted(list(counts.keys()))
    
    # 1. Plot Histogram
    sizes = [counts[r] for r in unique_regions]
    
    plt.figure(figsize=(10, 5))
    plt.bar(unique_regions, sizes, width=1.0, edgecolor='black')
    plt.xlabel('Codebook Region')
    plt.ylabel('Frequency (Frames)')
    plt.title('Codebook Region Size Distribution')
    plt.yscale('log')
    
    hist_path = os.path.join(out_dir, "region_distribution.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved distribution histogram to {hist_path}")
    
    # 2. Sample regions and frames
    sample_regions = random.sample(unique_regions, min(args.num_regions, len(unique_regions)))
    
    print(f"Visualizing {len(sample_regions)} randomly chosen regions...")
    for region in tqdm(sample_regions, desc="Process Regions"):
        region_dir = os.path.join(out_dir, f"region_{region}")
        os.makedirs(region_dir, exist_ok=True)
        
        # Get frame indices belonging to this region
        frame_indices = np.where(tokens == region)[0]
        
        n_samples = min(args.num_samples, len(frame_indices))
        selected_frames = np.random.choice(frame_indices, n_samples, replace=False)
        
        for sample_i, start_idx in enumerate(selected_frames, start=1):
            end_idx = min(start_idx + args.window, poses.shape[0])
            
            gen_poses = poses[start_idx:end_idx]
            gen_trans = trans[start_idx:end_idx].copy()
            
            mp4_out = os.path.join(region_dir, f"sample_{sample_i}.mp4")
            
            if len(gen_poses) < args.window:
                # If we hit the end of the poses array, skip or pad. 
                # (Create_codebook guarantees valid tokens have full windows, but just in case)
                pass
                
            render_animation_custom(gen_poses, gen_trans, mp4_out)

if __name__ == "__main__":
    main()
