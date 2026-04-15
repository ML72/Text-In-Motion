import os
import torch
import numpy as np
import imageio
import pyrender
import smplx
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

def render_animation(gen_poses, gen_trans, mp4_out):
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
