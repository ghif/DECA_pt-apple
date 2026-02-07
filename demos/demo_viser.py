# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

"""
This script provides an interactive 3D visualization of DECA reconstructions using the `viser` library.
Users can dial the shape and expressions of the 3D FLAME 2023 model.
"""

import os, sys
import cv2
import numpy as np
import argparse
import torch
import viser
import time
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets.datasets import TestData
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from decalib.utils.config import cfg as deca_cfg

def main(args):
    # 1. Setup Viser Server
    server = viser.ViserServer()
    server.scene.set_up_direction((0, 1, 0)) # FLAME up direction is Y
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, 0.0, 0.5)
        client.camera.look_at = (0.0, 0.0, 0.0)

    print(f"Viser server started at http://localhost:{server.get_port()}")
    loading_status = server.gui.add_button("Status: Loading DECA...", disabled=True)

    device = args.device
    use_detail = (args.mode == 'detail')
    
    # 2. Setup DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)

    # 3. Load Input Image
    testdata = TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, device=device)
    if len(testdata) == 0:
        print(f"No images found in {args.inputpath}")
        return

    # Use the first image for visualization
    i = 0
    image_item = testdata[i]
    images = image_item['image'].to(device)[None,...]
    image_name = testdata.imagepath_list[i]
    print(f"Loaded image: {image_name}")

    # 4. Initial Fast Reconstruction
    with torch.no_grad():
        codedict = deca.encode(images, use_detail=use_detail)
        # Note: we need the full opdict for initial faces/template
        opdict = deca.decode(codedict, return_vis=False, use_detail=use_detail)

    # Extract initial parameters for state initialization
    predicted_shape = codedict['shape'].clone()
    predicted_exp = codedict['exp'].clone()
    predicted_pose = codedict['pose'].clone()
    predicted_cam = codedict['cam'].clone()

    # State for controls
    state = {
        "shape_coeffs": [predicted_shape[0, i].item() for i in range(predicted_shape.shape[1])],
        "exp_coeffs": [predicted_exp[0, i].item() for i in range(predicted_exp.shape[1])],
        "global_rot": [predicted_pose[0, i].item() for i in range(3)], # Axis-angle
        "jaw_pose": [predicted_pose[0, i].item() for i in range(3, 6)], # Axis-angle
        "cam": [predicted_cam[0, i].item() for i in range(3)], # s, tx, ty
        "show_detail": use_detail, # Default to ON if in detail mode
        "show_texture": args.useTex, # Default to ON if useTex requested
        "texture_brightness": 1.15, # Default boost
        "use_projection": True 
    }

    # 5. GUI Setup
    loading_status.name = "Status: Setting up GUI..."
    
    # Show input image
    image_np = images[0].permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    server.gui.add_image(image=image_np, label="Input Image", format="jpeg")

    # Shared variables for update_mesh
    faces = deca.render.faces[0].cpu().numpy()
    dense_template = getattr(deca, 'dense_template', None)
    base_texture = None # Original extracted texture
    texture = None      # Current texture with brightness applied

    def update_mesh():
        nonlocal texture, base_texture
        print("Updating mesh...")
        with torch.no_grad():
            # 0. Update Shape
            new_shape = predicted_shape.clone()
            for idx, val in enumerate(state["shape_coeffs"]):
                new_shape[:, idx] = val
            codedict['shape'] = new_shape

            # 1. Update Expression
            new_exp = predicted_exp.clone()
            for idx, val in enumerate(state["exp_coeffs"]):
                new_exp[:, idx] = val 
            codedict['exp'] = new_exp
            
            # 2. Update Pose (Rotation + Jaw)
            new_pose = predicted_pose.clone()
            new_pose[0, :3] = torch.tensor(state["global_rot"]).to(device)
            new_pose[0, 3:] = torch.tensor(state["jaw_pose"]).to(device)
            codedict['pose'] = new_pose
            
            # 3. Update Camera (Scale + Translation)
            new_cam = predicted_cam.clone()
            new_cam[0, :] = torch.tensor(state["cam"]).to(device)
            codedict['cam'] = new_cam
            
            # 4. Decode
            print("  - Decoding...")
            current_opdict = deca.decode(codedict, return_vis=False, use_detail=(use_detail and state["show_detail"]))
            
            # 5. Handle fitted/projected view vs centered view
            # 5. Handle fitted/projected view vs centered view
            if state["use_projection"]:
                # In our new standardization, trans_verts is NDC space (Forehead at Y=-1, Top)
                # In Viser (Y=Up), this would be upside down. 
                # So we flip Y back for the 3D viewer.
                v_display = current_opdict['trans_verts'].clone()
                v_display[..., 1] = -v_display[..., 1]
                v_display[..., 2] = -v_display[..., 2] # Depth consistent with camera view
                display_verts = v_display[0].cpu().numpy()
            else:
                display_verts = current_opdict['verts'][0].cpu().numpy()

            if state["show_detail"] or state["show_texture"]:
                if state["show_texture"] and base_texture is None:
                    print("  - Extracting base texture...")
                    loading_status.name = "Status: Extracting Texture..."
                    # We need the full visdict for texture extraction
                    _, visdict = deca.decode(codedict, return_vis=True, use_detail=use_detail)
                    extracted = util.tensor2image(visdict['uv_texture_gt'][0])
                    # Fix BGR to RGB for Viser/Trimesh
                    base_texture = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
                    loading_status.name = "Status: Ready"

                # Apply brightness from state
                if base_texture is not None:
                    tex_f = base_texture.astype(np.float32) / 255.0
                    brightness = state["texture_brightness"]
                    # Boost brightness and contrast slightly to make it "dominant"
                    tex_f = np.clip(tex_f * brightness + (brightness - 1.0) * 0.2, 0, 1)
                    texture = (tex_f * 255).astype(np.uint8)

                print("  - Upsampling mesh...")
                normals = current_opdict['normals'][0].cpu().numpy()
                
                # If showing detail, use displacement map. Otherwise use zeros for flat skin.
                if use_detail and state["show_detail"]:
                    displacement_map = current_opdict['displacement_map'][0].cpu().numpy().squeeze()
                else:
                    displacement_map = np.zeros((deca.uv_size, deca.uv_size))
                
                # If showing texture, use extracted texture. Otherwise use grey.
                if state["show_texture"]:
                    v_colors = texture
                else:
                    v_colors = np.ones((deca.uv_size, deca.uv_size, 3)) * 0.7

                dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
                    current_opdict['verts'][0].cpu().numpy(), normals, faces, displacement_map, v_colors, dense_template
                )
                
                if state["use_projection"]:
                    s, tx, ty = state["cam"]
                    dense_vertices = s * (dense_vertices + np.array([tx, ty, 0.0]))

                print("  - Adding detailed skin mesh to scene...")
                server.scene.add_mesh_trimesh(
                    name="/face",
                    mesh=trimesh.Trimesh(vertices=dense_vertices, faces=dense_faces, vertex_colors=dense_colors)
                )
            else:
                print("  - Adding mesh to scene...")
                server.scene.add_mesh_trimesh(
                    name="/face",
                    mesh=trimesh.Trimesh(vertices=display_verts, faces=faces)
                )
        print("Mesh update complete.")

    # 5. GUI Controls
    loading_status.name = "Status: Ready"
    
    with server.gui.add_folder("Display Controls"):
        if use_detail:
            detail_checkbox = server.gui.add_checkbox("Show Detail Skin", initial_value=state["show_detail"])
            @detail_checkbox.on_update
            def _(_):
                state["show_detail"] = detail_checkbox.value
                update_mesh()
        
        texture_checkbox = server.gui.add_checkbox("Show UV Texture", initial_value=state["show_texture"])
        @texture_checkbox.on_update
        def _(_):
            state["show_texture"] = texture_checkbox.value
            update_mesh()
        
        brightness_slider = server.gui.add_slider("Texture Dominance", min=0.5, max=2.0, step=0.05, initial_value=state["texture_brightness"])
        @brightness_slider.on_update
        def _(_):
            state["texture_brightness"] = brightness_slider.value
            update_mesh()

        proj_checkbox = server.gui.add_checkbox("Show Fitted (Projected)", initial_value=state["use_projection"])
        @proj_checkbox.on_update
        def _(_):
            state["use_projection"] = proj_checkbox.value
            update_mesh()

        with server.gui.add_folder("Actions"):
            reset_all_button = server.gui.add_button("Reset to Predicted")
            @reset_all_button.on_click
            def _(_):
                # Restore state from predicted values
                state["shape_coeffs"] = [predicted_shape[0, i].item() for i in range(predicted_shape.shape[1])]
                state["exp_coeffs"] = [predicted_exp[0, i].item() for i in range(predicted_exp.shape[1])]
                state["global_rot"] = [predicted_pose[0, i].item() for i in range(3)]
                state["jaw_pose"] = [predicted_pose[0, i].item() for i in range(3, 6)]
                state["cam"] = [predicted_cam[0, i].item() for i in range(3)]
                
                # Sync sliders
                for i, slider in enumerate(shape_sliders):
                    slider.value = state["shape_coeffs"][i]
                for i, slider in enumerate(exp_sliders):
                    slider.value = state["exp_coeffs"][i]
                for i, slider in enumerate(rot_sliders):
                    slider.value = state["global_rot"][i]
                for i, slider in enumerate(jaw_sliders):
                    slider.value = state["jaw_pose"][i]
                for i, slider in enumerate(cam_sliders):
                    slider.value = state["cam"][i]

                update_mesh()

            random_exp_button = server.gui.add_button("Random Expression")
            @random_exp_button.on_click
            def _(_):
                for i in range(len(state["exp_coeffs"])):
                    val = np.random.uniform(-1.5, 1.5)
                    state["exp_coeffs"][i] = val
                    if i < len(exp_sliders):
                        exp_sliders[i].value = val
                update_mesh()

            reset_jaw_button = server.gui.add_button("Reset Jaw")
            @reset_jaw_button.on_click
            def _(_):
                for i in range(3):
                    val = predicted_pose[0, 3+i].item()
                    state["jaw_pose"][i] = val
                    jaw_sliders[i].value = val
                update_mesh()

        with server.gui.add_folder("Expression Presets"):
            btn_smile = server.gui.add_button("Smile")
            @btn_smile.on_click
            def _(_):
                # Simple heuristic for smile: increase some expression coefficients
                # This depends on the FLAME basis, but typically first few coeffs handle mouth
                for i in range(num_exp_sliders):
                    state["exp_coeffs"][i] = 0.0
                state["exp_coeffs"][0] = 2.0 # Often mouth open/smile-ish
                state["exp_coeffs"][1] = 1.0 
                for i, slider in enumerate(exp_sliders):
                    slider.value = state["exp_coeffs"][i]
                update_mesh()

            btn_surprised = server.gui.add_button("Surprised")
            @btn_surprised.on_click
            def _(_):
                for i in range(num_exp_sliders):
                    state["exp_coeffs"][i] = 0.0
                state["exp_coeffs"][3] = 2.0 # Brows up/Eye open
                state["jaw_pose"][0] = 0.5 # Jaw open
                for i, slider in enumerate(exp_sliders):
                    slider.value = state["exp_coeffs"][i]
                for i, slider in enumerate(jaw_sliders):
                    slider.value = state["jaw_pose"][i]
                update_mesh()

            btn_reset_exp = server.gui.add_button("Flat Face")
            @btn_reset_exp.on_click
            def _(_):
                for i in range(num_exp_sliders):
                    state["exp_coeffs"][i] = 0.0
                for i, slider in enumerate(exp_sliders):
                    slider.value = 0.0
                update_mesh()

            random_shape_button = server.gui.add_button("Randomize Shape")
            @random_shape_button.on_click
            def _(_):
                for i in range(len(state["shape_coeffs"])):
                    val = np.random.normal(0, 1.0)
                    state["shape_coeffs"][i] = val
                    if i < len(shape_sliders):
                        shape_sliders[i].value = val
                update_mesh()

    with server.gui.add_folder("Global Rotation (Axis-Angle)"):
        rot_labels = ["Rot X", "Rot Y", "Rot Z"]
        rot_sliders = []
        for i in range(3):
            slider = server.gui.add_slider(
                rot_labels[i], min=-3.14, max=3.14, step=0.01, initial_value=state["global_rot"][i]
            )
            rot_sliders.append(slider)
            def make_handler(idx):
                def handler(_):
                    state["global_rot"][idx] = rot_sliders[idx].value
                    update_mesh()
                return handler
            slider.on_update(make_handler(i))

    with server.gui.add_folder("Camera (Fitted Perspective)"):
        cam_labels = ["Scale", "Trans X", "Trans Y"]
        cam_mins = [0.1, -1.0, -1.0]
        cam_maxs = [20.0, 1.0, 1.0]
        cam_sliders = []
        for i in range(3):
            slider = server.gui.add_slider(
                cam_labels[i], min=cam_mins[i], max=cam_maxs[i], step=0.01, initial_value=state["cam"][i]
            )
            cam_sliders.append(slider)
            def make_cam_handler(idx):
                def handler(_):
                    state["cam"][idx] = cam_sliders[idx].value
                    update_mesh()
                return handler
            slider.on_update(make_cam_handler(i))

    with server.gui.add_folder("Jaw Rotation"):
        jaw_labels = ["Jaw Open/Close", "Jaw Swing (Y)", "Jaw Twist (Z)"]
        jaw_mins = [-0.1, -0.2, -0.2]
        jaw_maxs = [0.6, 0.2, 0.2]
        jaw_sliders = []
        for i in range(3):
            slider = server.gui.add_slider(
                jaw_labels[i], min=jaw_mins[i], max=jaw_maxs[i], step=0.01, initial_value=state["jaw_pose"][i]
            )
            jaw_sliders.append(slider)
            def make_jaw_handler(idx):
                def handler(_):
                    state["jaw_pose"][idx] = jaw_sliders[idx].value
                    update_mesh()
                return handler
            slider.on_update(make_jaw_handler(i))

    with server.gui.add_folder("Expression Coefficients"):
        exp_sliders = []
        # Show first 50 coeffs to avoid overwhelming UI, but full state is used
        num_exp_sliders = min(50, predicted_exp.shape[1])
        for i in range(num_exp_sliders):
            slider = server.gui.add_slider(
                f"Exp {i}", min=-3.0, max=3.0, step=0.1, initial_value=state["exp_coeffs"][i]
            )
            exp_sliders.append(slider)
            def make_exp_handler(idx):
                def handler(_):
                    state["exp_coeffs"][idx] = exp_sliders[idx].value
                    update_mesh()
                return handler
            slider.on_update(make_exp_handler(i))

    with server.gui.add_folder("Shape Coefficients"):
        shape_sliders = []
        # Show first 50 coeffs
        num_shape_sliders = min(50, predicted_shape.shape[1])
        for i in range(num_shape_sliders):
            slider = server.gui.add_slider(
                f"Shape {i}", min=-3.0, max=3.0, step=0.1, initial_value=state["shape_coeffs"][i]
            )
            shape_sliders.append(slider)
            def make_shape_handler(idx):
                def handler(_):
                    state["shape_coeffs"][idx] = shape_sliders[idx].value
                    update_mesh()
                return handler
            slider.on_update(make_shape_handler(i))

    # Initial Render
    update_mesh()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Interactive Visualization with Viser')
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str)
    parser.add_argument('--mode', default='coarse', choices=['coarse', 'detail'], type=str)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--rasterizer_type', default='standard', type=str)
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--detector', default='fan', type=str)
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    main(parser.parse_args())