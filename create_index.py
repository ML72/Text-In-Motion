import os
import pickle
import numpy as np
import glob
import json

def build_index(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    pkl_files.sort()
    
    all_poses = []
    all_trans = []
    all_file_indices = []
    all_frame_indices = []
    
    file_names = []
    
    print(f"Found {len(pkl_files)} motion files. Indexing...")
    
    for file_idx, pkl_file in enumerate(pkl_files):
        file_names.append(os.path.basename(pkl_file))
        
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        smpl_poses = data['smpl_poses']
        smpl_trans = data['smpl_trans']
        smpl_scaling = data['smpl_scaling']
        
        # AIST++ trans normally needs to be divided by scale to match standard translation scale
        smpl_trans = smpl_trans / smpl_scaling
        
        num_frames = smpl_poses.shape[0]
        
        all_poses.append(smpl_poses)
        all_trans.append(smpl_trans)
        
        all_file_indices.append(np.full((num_frames,), file_idx, dtype=np.int32))
        all_frame_indices.append(np.arange(num_frames, dtype=np.int32))
        
        if (file_idx + 1) % 50 == 0:
            print(f"Processed {file_idx + 1}/{len(pkl_files)} files...")
            
    if not all_poses:
        print("No poses found.")
        return
    
    print("Concatenating arrays...")
    all_poses = np.concatenate(all_poses, axis=0)      # (Total_F, 72)
    all_trans = np.concatenate(all_trans, axis=0)      # (Total_F, 3)
    all_file_indices = np.concatenate(all_file_indices, axis=0)
    all_frame_indices = np.concatenate(all_frame_indices, axis=0)
    
    out_pose_path = os.path.join(output_dir, "motion_index.npz")
    print(f"Saving index to {out_pose_path}")
    np.savez_compressed(
        out_pose_path,
        poses=all_poses,
        trans=all_trans,
        file_indices=all_file_indices,
        frame_indices=all_frame_indices
    )
    
    out_names_path = os.path.join(output_dir, "file_names.json")
    with open(out_names_path, "w") as f:
        json.dump(file_names, f, indent=4)
        
    print(f"Done! Total frames indexed: {all_poses.shape[0]}")

if __name__ == "__main__":
    input_directory = os.path.join("data", "motions")
    output_directory = os.path.join("data", "index")
    build_index(input_directory, output_directory)
