import os
import argparse
import random
import pickle
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import smplx
import torch
import pyrender
import trimesh
import trimesh.transformations as tra

from generate_sample import generate_motion, dijkstra_shortest_path

def ensure_dirs():
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

def levenshtein_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    if n == 0: return m
    if m == 0: return n
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            d[i, j] = min(
                d[i-1, j] + 1,     # deletion
                d[i, j-1] + 1,     # insertion
                d[i-1, j-1] + cost # substitution
            )
    return d[n, m]

def extract_ngrams(sequence, n):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def get_aist_transitions_and_energy(num_transitions):
    index_path = os.path.join("data", "index", "motion_index.npz")
    codebook_path = os.path.join("data", "index", "codebook.npz")
    
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    
    codebook_data = np.load(codebook_path)
    tokens = codebook_data['tokens']
    
    valid_mask = (file_indices[:-1] == file_indices[1:])
    valid_mask = np.append(valid_mask, False)
    
    # Calculate kinetic energy (velocities)
    pose_vel = np.zeros_like(poses)
    trans_vel = np.zeros_like(trans)
    pose_vel[:-1][valid_mask[:-1]] = poses[1:][valid_mask[:-1]] - poses[:-1][valid_mask[:-1]]
    trans_vel[:-1][valid_mask[:-1]] = trans[1:][valid_mask[:-1]] - trans[:-1][valid_mask[:-1]]
    
    kinetic_energy = np.sum(pose_vel**2, axis=1) + np.sum(trans_vel**2, axis=1)
    
    # Extract sequences of valid regions
    sequences = []
    current_seq = []
    for i in range(len(tokens)):
        r = tokens[i]
        if r != -1:
            current_seq.append(r)
        if not valid_mask[i] or r == -1:
            if len(current_seq) > 1:
                sequences.append(current_seq)
            current_seq = []
            
    # Sample random sequences until we hit num_transitions
    sampled_seqs = []
    trans_count = 0
    sampled_energies = []
    
    random.shuffle(sequences)
    for seq in sequences:
        if trans_count >= num_transitions:
            break
        sampled_seqs.append(seq)
        trans_count += len(seq) - 1
        
    # Get 5 full contiguous sequences of kinetic energy for Plot 3
    aist_eval_energies = []
    unique_files = np.unique(file_indices)
    random.shuffle(unique_files)
    for uf in unique_files:
        if len(aist_eval_energies) >= 5: break
        idx = np.where((file_indices == uf) & valid_mask)[0]
        if len(idx) > 600:  # Valid length
            aist_eval_energies.append(kinetic_energy[idx][:600])

    return sampled_seqs, aist_eval_energies

def plot_upset(ax_bars, ax_matrix, aist_n, auto_n, text_n, title):
    set_names = ['AIST++', 'Auto', 'Text']
    
    only_aist = len(aist_n - auto_n - text_n)
    only_auto = len(auto_n - aist_n - text_n)
    only_text = len(text_n - aist_n - auto_n)
    a_and_a = len((aist_n & auto_n) - text_n)
    a_and_t = len((aist_n & text_n) - auto_n)
    u_and_t = len((auto_n & text_n) - aist_n)
    all_3 = len(aist_n & auto_n & text_n)
    
    intersections = [
        ((1,0,0), only_aist),
        ((0,1,0), only_auto),
        ((0,0,1), only_text),
        ((1,1,0), a_and_a),
        ((1,0,1), a_and_t),
        ((0,1,1), u_and_t),
        ((1,1,1), all_3)
    ]
    intersections.sort(key=lambda x: x[1], reverse=True)
    
    sizes = [x[1] for x in intersections]
    matrices = [x[0] for x in intersections]
    
    x = np.arange(len(sizes))
    colors = plt.cm.tab10.colors[:len(sizes)]
    ax_bars.bar(x, sizes, color=colors)
    ax_bars.set_title(title, pad=20)
    ax_bars.set_ylabel("Intersection Size")
    ax_bars.set_xticks([])
    
    # Hide borders
    for spine in ['top', 'right', 'bottom']:
        ax_bars.spines[spine].set_visible(False)
        
    ax_matrix.set_ylim(-0.5, len(set_names)-0.5)
    ax_matrix.set_xlim(-0.5, len(sizes)-0.5)
    ax_matrix.set_yticks(np.arange(len(set_names)))
    ax_matrix.set_yticklabels(set_names)
    ax_matrix.set_xticks([])
    
    for i, combination in enumerate(matrices):
        y_coords = []
        for r, in_set in enumerate(combination):
            if in_set:
                ax_matrix.scatter(i, r, color='black', s=100, zorder=5)
                y_coords.append(r)
            else:
                ax_matrix.scatter(i, r, color='lightgray', s=100, zorder=5)
        if len(y_coords) > 1:
            ax_matrix.plot([i, i], [min(y_coords), max(y_coords)], color='black', lw=2, zorder=4)
            
    for spine in ['top', 'right', 'bottom', 'left']:
        ax_matrix.spines[spine].set_visible(False)

def run_part_1():
    print("Running Part 1: Stylistic & Art-Side Evaluation")
    
    # 1. Gather AIST++ transitions
    print("Gathering Original AIST++ data...")
    aist_seqs, aist_energy = get_aist_transitions_and_energy(5000)
    aist_ngrams_3 = set(ng for seq in aist_seqs for ng in extract_ngrams(seq, 3))
    aist_ngrams_4 = set(ng for seq in aist_seqs for ng in extract_ngrams(seq, 4))
    
    # 2. Gather Autonomous Mode transitions
    print("Generating Autonomous Mode sequences...")
    auto_seqs = []
    auto_eval_energies = []
    auto_trans_count = 0
    idx = 0

    with tqdm(total=5000, desc="Autonomous") as pbar:
        while auto_trans_count < 5000:
            run_data, g_poses, g_trans = generate_motion(
                num_frames=720, 
                run_name=f"auto_{idx}", 
                gender='neutral', 
                physics_algorithms_on=False, 
                render_video=False, 
                verbose=False
            )
            regions = [int(x) for x in run_data["executed_dna"].split(',') if x.strip()]
            auto_seqs.append(regions)
            
            added_tc = max(0, len(regions) - 1)
            auto_trans_count += added_tc
            pbar.update(added_tc)
            
            if len(auto_eval_energies) < 5 and len(g_poses) > 600:
                ke = np.sum(np.diff(g_poses[:601], axis=0)**2, axis=1) + np.sum(np.diff(g_trans[:601], axis=0)**2, axis=1)
                auto_eval_energies.append(ke)
                
            idx += 1
        
    # 3. Gather Text-Guided Mode transitions
    print("Generating Text-Guided Mode sequences...")
    txt_files = glob.glob('data/eval_text/*.txt')
    lines = []
    for tf in txt_files:
        with open(tf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    random.shuffle(lines)
    
    text_seqs = []
    text_eval_energies = []
    text_trans_count = 0
    txt_all_en_list = []
    idx = 0
    
    with tqdm(total=5000, desc="Text-Guided") as pbar:
        while text_trans_count < 5000 and idx < len(lines):
            line = lines[idx]
            idx += 1
            if len(text_eval_energies) < 5 and len(line) >= 20:
                line_to_use = line[:20]
            else:
                line_to_use = line
                
            # Run without rendering or physics filtering (for speed of dataset collection)
            run_data, g_poses, g_trans = generate_motion(num_frames=0, run_name=f"text_{idx}", input_text=line_to_use, render_video=False, physics_algorithms_on=False, verbose=False)
            regions = [int(x) for x in run_data["executed_dna"].split(',') if x.strip()]
            text_seqs.append(regions)
            
            added_tc = max(0, len(regions) - 1)
            text_trans_count += added_tc
            pbar.update(added_tc)
            
            # extract whole curves up to 5
            if len(g_poses) > 1:
                ke_all = np.sum(np.diff(g_poses, axis=0)**2, axis=1) + np.sum(np.diff(g_trans, axis=0)**2, axis=1)
                txt_all_en_list.append(ke_all)
            
            if len(text_eval_energies) < 5 and len(line) >= 20 and len(g_poses) > 600:
                ke = np.sum(np.diff(g_poses[:601], axis=0)**2, axis=1) + np.sum(np.diff(g_trans[:601], axis=0)**2, axis=1)
                text_eval_energies.append(ke)
                
    # Metric: Choreographic N-Gram Novelty
    text_ngrams_3 = set(ng for seq in text_seqs for ng in extract_ngrams(seq, 3))
    text_ngrams_4 = set(ng for seq in text_seqs for ng in extract_ngrams(seq, 4))
    
    novel_3 = text_ngrams_3 - aist_ngrams_3
    novel_4 = text_ngrams_4 - aist_ngrams_4
    
    perc_novel_3 = len(novel_3) / max(1, len(text_ngrams_3)) * 100
    perc_novel_4 = len(novel_4) / max(1, len(text_ngrams_4)) * 100
    
    # Metric: Sequence Edit Distance (determinism vs diversity)
    # pair 100 random text seqs
    distances = []
    sub_text_seqs = random.sample(text_seqs, min(100, len(text_seqs)))
    for i in range(len(sub_text_seqs)):
        for j in range(i+1, len(sub_text_seqs)):
            distances.append(levenshtein_distance(sub_text_seqs[i], sub_text_seqs[j]))
            
    avg_edit_dist = np.mean(distances) if distances else 0.0
    
    aist_all_en = np.concatenate(aist_energy) if len(aist_energy) else [0]
    txt_all_en = np.concatenate(txt_all_en_list) if len(txt_all_en_list) else [0]
    
    # Save metrics
    with open('results/metrics/part_1.txt', 'w') as f:
        f.write(f"Choreographic 3-Gram Novelty: {perc_novel_3:.2f}%\n")
        f.write(f"Choreographic 4-Gram Novelty: {perc_novel_4:.2f}%\n")
        f.write(f"Average Sequence Edit Distance (100 samples): {avg_edit_dist:.2f}\n")
        f.write(f"AIST++ Mean Kinetic Energy: {np.mean(aist_all_en):.4f}\n")
        f.write(f"Text-Guided Mean Kinetic Energy: {np.mean(txt_all_en):.4f}\n")
        
    print("Saving plots...")
    # Plot 1: Codebook Identity Grids
    index_path = os.path.join("data", "index", "motion_index.npz")
    codebook_path = os.path.join("data", "index", "codebook.npz")
    poses = np.load(index_path)['poses']
    tokens = np.load(codebook_path)['tokens']
    
    fig_grid = plt.figure(figsize=(20, 10))
    fig_grid.suptitle("Codebook Identity Grids (Regions 0-4)\nMale (Blue) | Female (Pink) | Neutral (Purple)", fontsize=16)
    
    # Render setup
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    scene = pyrender.Scene(ambient_light=(0.6, 0.6, 0.6), bg_color=[1.0, 1.0, 1.0, 1.0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    key_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    kl_pose = np.eye(4); kl_pose[:3, :3] = tra.euler_matrix(np.radians(-45), np.radians(45), 0)[:3, :3]
    scene.add(key_light, pose=kl_pose)
    fill_light = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=1.2)
    fl_pose = np.eye(4); fl_pose[:3, :3] = tra.euler_matrix(np.radians(-30), np.radians(-45), 0)[:3, :3]
    scene.add(fill_light, pose=fl_pose)
    back_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    bl_pose = np.eye(4); bl_pose[:3, :3] = tra.euler_matrix(np.radians(-135), np.radians(0), 0)[:3, :3]
    scene.add(back_light, pose=bl_pose)

    cam_pose = np.eye(4) @ tra.rotation_matrix(np.radians(-15 * np.pi / 180), [1, 0, 0])
    cam_pose[1, 3] = 0.1
    cam_pose[2, 3] = 2.2
    cam_node = scene.add(camera, pose=cam_pose)

    # Base models
    smpl_models = {
        'male': smplx.create("models", model_type='smpl', gender='male', batch_size=1),
        'female': smplx.create("models", model_type='smpl', gender='female', batch_size=1),
        'neutral': smplx.create("models", model_type='smpl', gender='neutral', batch_size=1)
    }
    mats = {
        'male': pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.2, 0.4, 0.8, 1.0], metallicFactor=0.2, roughnessFactor=0.6, doubleSided=True),
        'female': pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.9, 0.3, 0.6, 1.0], metallicFactor=0.2, roughnessFactor=0.6, doubleSided=True),
        'neutral': pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.5, 0.2, 0.8, 1.0], metallicFactor=0.2, roughnessFactor=0.6, doubleSided=True)
    }
    renderer = pyrender.OffscreenRenderer(viewport_width=300, viewport_height=300)
    
    for r in tqdm(range(5), desc="Rendering Codebook Poses"):
        region_idx = np.where(tokens == r)[0]
        if len(region_idx) == 0: continue
        sampled_i = np.random.choice(region_idx, size=min(9, len(region_idx)), replace=False)
        
        genders = ['male'] * 3 + ['female'] * 3 + ['neutral'] * 3
        np.random.shuffle(genders)
        
        # Add label in the first spot
        ax_label = fig_grid.add_subplot(5, 10, r*10 + 1)
        ax_label.text(0.5, 0.5, f"Region {r}", ha='center', va='center', fontsize=14)
        ax_label.set_axis_off()
        
        for c, i in enumerate(sampled_i):
            gender = genders[c]
            
            p = torch.tensor(poses[i:i+1], dtype=torch.float32)
            out = smpl_models[gender](global_orient=p[:,:3], body_pose=p[:,3:], transl=torch.zeros((1,3)))
            vertices = out.vertices.detach().cpu().numpy().squeeze()
            faces = smpl_models[gender].faces
            
            mesh = trimesh.Trimesh(vertices, faces)
            mesh_node = pyrender.Mesh.from_trimesh(mesh, material=mats[gender], smooth=True)
            node = scene.add(mesh_node)
            
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(node)
            
            ax = fig_grid.add_subplot(5, 10, r*10 + c + 2)
            ax.imshow(color)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

    renderer.delete()
    plt.tight_layout()
    plt.savefig("results/plots/plot_1_codebook_grid.png", dpi=150)
    plt.close()
    
    # Plot 2: UpSet Plot
    auto_ngrams_3 = set(ng for seq in auto_seqs for ng in extract_ngrams(seq, 3))
    auto_ngrams_4 = set(ng for seq in auto_seqs for ng in extract_ngrams(seq, 4))
    
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.1)
    
    ax_bar_3 = fig.add_subplot(gs[0, 0])
    ax_mat_3 = fig.add_subplot(gs[1, 0], sharex=ax_bar_3)
    plot_upset(ax_bar_3, ax_mat_3, aist_ngrams_3, auto_ngrams_3, text_ngrams_3, "Unique 3-Gram Sequence Overlap")
    
    ax_bar_4 = fig.add_subplot(gs[0, 1])
    ax_mat_4 = fig.add_subplot(gs[1, 1], sharex=ax_bar_4)
    plot_upset(ax_bar_4, ax_mat_4, aist_ngrams_4, auto_ngrams_4, text_ngrams_4, "Unique 4-Gram Sequence Overlap")
    
    plt.savefig("results/plots/plot_2_phrase_overlap.png", dpi=150)
    plt.close()
    
    # Plot 3: Rhythmic Envelope (Time Series)
    plt.figure(figsize=(12, 6))
    
    colors = {'AIST++': '#1f77b4', 'Autonomous': '#ff7f0e', 'Text-Guided': '#2ca02c'}
    
    def plot_lines(arr_list, label, color):
        for i, arr in enumerate(arr_list):
            if len(arr) < 600: continue
            custom_label = label if i == 0 else ""
            plt.plot(np.arange(600), arr[:600], color=color, linewidth=1.2, alpha=0.7, label=custom_label)

    plot_lines(aist_energy, 'AIST++', colors['AIST++'])
    plot_lines(auto_eval_energies, 'Autonomous', colors['Autonomous'])
    plot_lines(text_eval_energies, 'Text-Guided', colors['Text-Guided'])
    
    plt.yscale('log')
    plt.title("Kinetic Signature (Kinetic Energy over Time)")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Kinetic Energy (Squared Velocity Sum)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/plot_3_rhythmic_envelope.png", dpi=150)
    plt.close()

def run_part_2():
    print("Running Part 2: Structural Graph Traversability")
    graph_path = os.path.join("data", "index", "plausibility_graph.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    num_nodes = 256
    num_possible_edges = num_nodes * (num_nodes - 1)
    num_edges = sum(len(graph.get(k, {})) for k in range(num_nodes))
    density = num_edges / num_possible_edges if num_possible_edges > 0 else 0
    
    alphanumeric = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alpha_regions = [ord(c) for c in alphanumeric]
    
    failures = 0
    total_alpha_pairs = 0
    for i in tqdm(alpha_regions, desc="Alphanumeric Regions"):
        for j in alpha_regions:
            if i != j:
                total_alpha_pairs += 1
                path = dijkstra_shortest_path(graph, i, j)
                if not path:
                    failures += 1
                    
    fail_rate = (failures / total_alpha_pairs) * 100 if total_alpha_pairs > 0 else 0
    
    # Average Dijkstra bridges
    sampled_pairs = []
    nodes = list(range(256))
    for _ in range(10000):
        n1, n2 = random.sample(nodes, 2)
        sampled_pairs.append((n1, n2))
        
    bridges_dist = []
    # Using progress bar
    for n1, n2 in tqdm(sampled_pairs, desc="Dijkstra Sampling"):
        path = dijkstra_shortest_path(graph, n1, n2)
        if path:
            # path is e.g. [inter1, inter2, target].
            # original prompt length to add was intermediate bridges. 
            # if direct, path is [target], length 1. So bridges = len(path) - 1
            bridges_dist.append(len(path) - 1)
            
    avg_bridges = np.mean(bridges_dist) if bridges_dist else 0
    
    with open('results/metrics/part_2.txt', 'w') as f:
        f.write(f"Graph Density: {density:.6f}\n")
        f.write(f"Alphanumeric Pathfinding Failure Rate: {fail_rate:.2f}%\n")
        f.write(f"Average Dijkstra Bridges per Region Transition: {avg_bridges:.2f}\n")
        
    # Plot 4: Bridge Density per Region Transition
    plt.figure(figsize=(10, 6))
    counts = np.bincount(bridges_dist)
    x = np.arange(len(counts))
    plt.bar(x, counts / np.sum(counts), color='#9467bd')
    plt.title("Bridge Density per Region Transition")
    plt.xlabel("Number of Intermediate Bridges Injected")
    plt.ylabel("Relative Frequency")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig("results/plots/plot_4_bridge_density.png", dpi=150)
    plt.close()

def compute_mesh_stats(poses_list, trans_list):
    if len(poses_list) == 0:
        return {}
        
    body_model = smplx.create("models", model_type='smpl', gender='neutral', batch_size=1)
    
    all_fpe = []
    skating_errors = []
    jerks = []
    
    # Analyze essentially as sequences
    for i in tqdm(range(len(poses_list)), desc="Mesh Stats"):
        poses = torch.tensor(poses_list[i], dtype=torch.float32)
        trans = torch.tensor(trans_list[i], dtype=torch.float32)
        
        # Batch size needs to match
        local_body = smplx.create("models", model_type='smpl', gender='neutral', batch_size=poses.shape[0])
        out = local_body(global_orient=poses[:,:3], body_pose=poses[:,3:], transl=trans)
        
        v = out.vertices.detach().numpy()
        j = out.joints.detach().numpy()
        
        # FPE: mesh clipping below Y=0
        min_y = np.min(v[:, :, 1], axis=1)
        fpe_clip = np.clip(-min_y, 0, None)
        all_fpe.extend(fpe_clip)
        
        # FSR: horizontal foot slippage during planted states (Y < 2cm)
        foot_joints = j[:, [7, 8, 10, 11], :]
        for f in range(4):
            vel = np.linalg.norm(np.diff(foot_joints[:, f, :], axis=0), axis=-1)
            # Find planted frames. Approximate Y < 2cm
            # Note: joints are inside the foot volume, typically ~4-8cm above the floor.
            # We relax the threshold to 0.08m (8cm) to reliably capture planted joints, 
            # since a true y_height of 2cm strictly applies to surface meshes, not bone joints.
            y_heights = foot_joints[:-1, f, 1]
            planted = y_heights < 0.08
            if np.any(planted):
                # Calculate horizontal sliding in planted frame
                xz_vel = np.linalg.norm(np.diff(foot_joints[:, f, [0, 2]], axis=0), axis=-1)
                skating_errors.extend(xz_vel[planted])
                
        # Jerk (3rd derivative)
        if len(trans) > 3:
            vel = np.diff(trans.numpy(), axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            j_mag = np.linalg.norm(jerk, axis=-1)
            jerks.extend(j_mag)
            
    return {
        "FPE": all_fpe,
        "Skating": skating_errors,
        "Jerk": jerks
    }

def print_stats(name, data_dict, f):
    f.write(f"--- {name} ---\n")
    for k, v in data_dict.items():
        if len(v) == 0:
            f.write(f"  {k}: No data\n")
            continue
        v_np = np.array(v)
        f.write(f"  {k} - Mean: {np.mean(v_np):.5f}, Median: {np.median(v_np):.5f}, Min: {np.min(v_np):.5f}, Max: {np.max(v_np):.5f}, SD: {np.std(v_np):.5f}\n")

def run_part_3():
    print("Running Part 3: System Polish & Quality Assurance")
    
    # 1. Gather original AIST++ 10 sequences x 1000 frames
    index_path = os.path.join("data", "index", "motion_index.npz")
    index_data = np.load(index_path)
    poses = index_data['poses']
    trans = index_data['trans']
    file_indices = index_data['file_indices']
    
    valid_mask = (file_indices[:-1] == file_indices[1:])
    valid_mask = np.append(valid_mask, False)
    
    aist_poses = []
    aist_trans = []
    
    v_indices = np.where(valid_mask)[0]
    for _ in range(10):
        start = random.choice(v_indices[:-1000])
        if file_indices[start] == file_indices[start+999] and np.all(valid_mask[start:start+1000]):
            aist_poses.append(poses[start:start+1000])
            aist_trans.append(trans[start:start+1000])
            
    aist_stats = compute_mesh_stats(aist_poses, aist_trans)
    
    # 2. Autonomous W/O Physics
    print("Generating Autonomous Mode sequences Without Physics...")
    auto_np_poses = []
    auto_np_trans = []
    for i in tqdm(range(10), desc="Bench NP"):
        _, p, t = generate_motion(1000, f"bench_np_{i}", gender='neutral', physics_algorithms_on=False, render_video=False, verbose=False)
        auto_np_poses.append(p)
        auto_np_trans.append(t)
    auto_np_stats = compute_mesh_stats(auto_np_poses, auto_np_trans)
    
    # 3. Autonomous WITH Physics
    print("Generating Autonomous Mode sequences With Physics...")
    auto_wp_poses = []
    auto_wp_trans = []
    for i in tqdm(range(10), desc="Bench WP"):
        _, p, t = generate_motion(1000, f"bench_wp_{i}", gender='neutral', physics_algorithms_on=True, render_video=False, verbose=False)
        auto_wp_poses.append(p)
        auto_wp_trans.append(t)
    auto_wp_stats = compute_mesh_stats(auto_wp_poses, auto_wp_trans)
    
    with open('results/metrics/part_3.txt', 'w') as f:
        print_stats("AIST++ Original", aist_stats, f)
        print_stats("Autonomous Without Physics", auto_np_stats, f)
        print_stats("Autonomous With Physics", auto_wp_stats, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, choices=[1, 2, 3], help="Run a specific part of the analysis (1, 2, or 3).")
    args = parser.parse_args()
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    ensure_dirs()
    
    if args.part == 1 or args.part is None:
        run_part_1()
    
    if args.part == 2 or args.part is None:
        run_part_2()
        
    if args.part == 3 or args.part is None:
        run_part_3()

if __name__ == "__main__":
    main()
