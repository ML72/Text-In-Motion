import os
import torch
import numpy as np
import imageio
import pyrender
import smplx
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

def render_animation(gen_poses, gen_trans, mp4_out, gender='neutral'):
    print("Rendering video...")
    try:
        body_model = smplx.create("models", model_type='smpl', gender=gender, batch_size=1)
    except Exception as e:
        print("Visualization skipped. Could not load SMPL models.")
        print(f"Error: {e}")
        return

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    scene = pyrender.Scene(ambient_light=(0.6, 0.6, 0.6), bg_color=[1.0, 1.0, 1.0, 1.0])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)  # Wider FOV to zoom out
    
    # 3-point lighting setup for better visual appeal
    # Key light
    key_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    key_light_pose = np.eye(4)
    key_light_pose[:3, :3] = tra.euler_matrix(np.radians(-45), np.radians(45), 0)[:3, :3]
    scene.add(key_light, pose=key_light_pose)
    
    # Fill light
    fill_light = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=1.2)
    fill_light_pose = np.eye(4)
    fill_light_pose[:3, :3] = tra.euler_matrix(np.radians(-30), np.radians(-45), 0)[:3, :3]
    scene.add(fill_light, pose=fill_light_pose)
    
    # Back light
    back_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    back_light_pose = np.eye(4)
    back_light_pose[:3, :3] = tra.euler_matrix(np.radians(-135), np.radians(0), 0)[:3, :3]
    scene.add(back_light, pose=back_light_pose)

    rotation = tra.rotation_matrix(np.radians(-20 * np.pi / 180), [1, 0, 0])
    camera_pose = np.eye(4) @ rotation
    camera_pose[1, 3] = 1.0
    camera_pose[2, 3] = 3.0
    cam_node = scene.add(camera, pose=camera_pose)

    # Add ground guide lines for better motion perception
    try:
        import trimesh.creation
        
        # Add a solid floor to receive shadows
        floor_mesh = trimesh.creation.box(extents=[40.0, 0.02, 40.0])
        floor_mesh.apply_translation([0, -0.01, 0])
        floor_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.95, 0.95, 0.95, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.8
        )
        scene.add(pyrender.Mesh.from_trimesh(floor_mesh, material=floor_material, smooth=False))

        sm_cyls = []
        for x in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.006, height=20.0)
            # Cylinder is along Z by default, just translate along X
            cyl.apply_translation([x, 0.01, 0])
            sm_cyls.append(cyl)
        for z in np.arange(-10, 11, 1):
            cyl = trimesh.creation.cylinder(radius=0.006, height=20.0)
            # Rotate 90 degrees around Y to point along X, then translate along Z
            cyl.apply_transform(tra.rotation_matrix(np.pi/2, [0, 1, 0]))
            cyl.apply_translation([0, 0.01, z])
            sm_cyls.append(cyl)
        grid_mesh = trimesh.util.concatenate(sm_cyls)
        
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.65, 0.65, 0.65, 1.0], 
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

        dancer_x = gen_trans[i][0]
        dancer_y = gen_trans[i][1]
        dancer_z = gen_trans[i][2]
        
        dx = dancer_x - 0.0
        dz = dancer_z - 3.0
        # Camera looks down -Z in its local frame.
        # R_y * [0, 0, -1]^T = [-sin(theta), 0, -cos(theta)]^T
        # We want this to be proportional to [dx, 0, dz]
        # so sin(theta) = -dx, cos(theta) = -dz
        pan_angle = np.arctan2(-dx, -dz)
        
        # Keep vertical angle constant to prevent jitter (0 degrees centers around the pelvis)
        tilt_angle = np.radians(0)
        
        pan_rotation = tra.rotation_matrix(pan_angle, [0, 1, 0])
        tilt_rotation = tra.rotation_matrix(tilt_angle, [1, 0, 0])
        
        new_camera_pose = np.eye(4)
        # Apply pan (yaw) then tilt (pitch) 
        new_camera_pose = pan_rotation @ tilt_rotation
        new_camera_pose[0, 3] = 0.0
        new_camera_pose[1, 3] = 1.0
        new_camera_pose[2, 3] = 3.0
        scene.set_pose(cam_node, pose=new_camera_pose)
        
        # In a 3-point light setup we don't necessarily want the lights to follow the camera,
        # but if we do, we could update them here. For a studio look, stationary lights usually look better.
        # But we'll leave them stationary.

        global_orient = pose[:, :3]
        body_pose = pose[:, 3:]

        output = body_model(global_orient=global_orient,
                            body_pose=body_pose,
                            transl=trans)

        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = body_model.faces

        # Determine color based on gender
        if gender == 'neutral':
            char_color = [0.5, 0.2, 0.8, 1.0] # Purple
        elif gender == 'male':
            char_color = [0.2, 0.4, 0.8, 1.0] # Blue
        elif gender == 'female':
            char_color = [0.9, 0.3, 0.6, 1.0] # Pink
        else:
            char_color = [0.2, 0.4, 0.8, 1.0] # Fallback

        # Create a professional looking material for the character
        character_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=char_color,
            metallicFactor=0.2,
            roughnessFactor=0.6,
            doubleSided=True
        )

        mesh = trimesh.Trimesh(vertices, faces)
        mesh_node = pyrender.Mesh.from_trimesh(mesh, material=character_mat, smooth=True)

        node = scene.add(mesh_node)
        
        # Render with shadows
        flags = pyrender.RenderFlags.RGBA
        color, _ = renderer.render(scene, flags=flags)
        
        # Drop alpha channel if present for mp4 writer
        if color.shape[-1] == 4:
            color = color[..., :3]
            
        writer.append_data(color)
        scene.remove_node(node)

    writer.close()
    renderer.delete()
    print(f"Saved visualization to {mp4_out}")
