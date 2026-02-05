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
Users can:
1. Load a face image and reconstruct the 3D face.
2. Interactively modify expression parameters (jaw pose, expression coefficients) via a web-based GUI.
3. View the updated 3D face mesh in real-time.

Usage:
    python demos/demo_viser.py -i <path_to_image>
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
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from decalib.utils.config import cfg as deca_cfg

def main(args):
    # 1. Setup Viser Server first for fast perceived startup
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
    
    # 2. Setup DECA (slow)
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device, use_detail=use_detail)

    # 3. Load Input Image
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, device=device)
    if len(testdata) == 0:
        print(f"No images found in {args.inputpath}")
        return

    # Use the first image for visualization
    i = 0
    name = testdata[i]['imagename']
    images = testdata[i]['image'].to(device)[None,...]
    original_image_path = testdata.imagepath_list[i]
    print(f"Loaded image: {original_image_path}")

    # 4. Initial Fast Reconstruction (coarse only)
    with torch.no_grad():
        codedict = deca.encode(images, use_detail=use_detail)
        opdict = deca.decode(codedict, return_vis=False, use_detail=use_detail)

    # Extract initial parameters
    # shape: [1, 100], exp: [1, 50], pose: [1, 6] (3 global + 3 jaw)
    initial_exp = codedict['exp'].clone()
    initial_pose = codedict['pose'].clone()
    
    # Get faces and template for detailed mesh rendering
    faces = deca.render.faces[0].cpu().numpy()
    dense_template = getattr(deca, 'dense_template', None)
    
    # Lazy texture loading
    texture = None 

    # State for controls
    state = {
        "jaw_open": 0.0,
        "exp_coeffs": [0.0] * 10, # Control first 10 expression coefficients
        "global_rot": [0.0, 0.0, 0.0], # Pitch, Yaw, Roll
        "show_detail": False # Default to False for faster initial render
    }

    def update_mesh():
        nonlocal texture
        # Update codedict based on state
        with torch.no_grad():
            # 1. Update Expression
            new_exp = initial_exp.clone()
            for idx, val in enumerate(state["exp_coeffs"]):
                new_exp[:, idx] = val 
            codedict['exp'] = new_exp
            
            # 2. Update Jaw Pose
            euler_jaw = torch.zeros((1, 3)).to(device)
            euler_jaw[:, 0] = state["jaw_open"] # Jaw open/close
            jaw_pose = batch_euler2axis(deg2rad(euler_jaw))
            codedict['pose'][:, 3:] = jaw_pose
            
            # Decode to get new verts and displacement
            # Note: return_vis=False is enough if we already have the texture
            opdict = deca.decode(codedict, return_vis=False, use_detail=(use_detail and state["show_detail"]))
            
            if use_detail and state["show_detail"]:
                # Lazy-load texture if needed
                if texture is None:
                    print("Extracting texture for detailed skin...")
                    loading_status.name = "Status: Extracting Texture..."
                    with torch.no_grad():
                        _, visdict = deca.decode(codedict, return_vis=True, use_detail=True)
                        texture = util.tensor2image(visdict['uv_texture_gt'][0])
                    loading_status.name = "Status: Ready"

                # Upsample mesh for details
                vertices = opdict['verts'][0].cpu().numpy()
                normals = opdict['normals'][0].cpu().numpy()
                displacement_map = opdict['displacement_map'][0].cpu().numpy().squeeze()
                
                dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
                    vertices, normals, faces, displacement_map, texture, dense_template
                )
                
                # Update mesh in Viser using trimesh to support vertex colors
                mesh = trimesh.Trimesh(
                    vertices=dense_vertices,
                    faces=dense_faces,
                    vertex_colors=dense_colors
                )
                server.scene.add_mesh_trimesh(
                    name="/face",
                    mesh=mesh
                )
            else:
                # Show coarse mesh (no texture for simplicity if not detail)
                verts = opdict['verts'][0].cpu().numpy()
                mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=faces
                )
                server.scene.add_mesh_trimesh(
                    name="/face",
                    mesh=mesh
                )

    # 5. Add GUI Controls
    with server.gui.add_folder("Display Controls"):
        if use_detail:
            detail_checkbox = server.gui.add_checkbox("Show Detail Skin", initial_value=False)
            @detail_checkbox.on_update
            def _(_):
                state["show_detail"] = detail_checkbox.value
                update_mesh()
        else:
            server.gui.add_button("Mode: Coarse Only", disabled=True)

    with server.gui.add_folder("Expression Controls"):
        # Jaw Control
        jaw_slider = server.gui.add_slider(
            "Jaw Open",
            min=0.0,
            max=60.0,
            step=1.0,
            initial_value=0.0
        )
        
        @jaw_slider.on_update
        def _(_) -> None:
            state["jaw_open"] = jaw_slider.value
            update_mesh()

        # Expression Coefficients
        exp_sliders = []
        for i in range(10): # First 10 coeffs
            slider = server.gui.add_slider(
                f"Exp {i}",
                min=-3.0,
                max=3.0,
                step=0.1,
                initial_value=initial_exp[0, i].item()
            )
            exp_sliders.append(slider)
            
            def make_handler(idx):
                def handler(_) -> None:
                    state["exp_coeffs"][idx] = exp_sliders[idx].value
                    update_mesh()
                return handler
            
            slider.on_update(make_handler(i))

    # Initial Render
    update_mesh()

    # Keep alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Interactive Visualization with Viser')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to the test data, can be image folder, image path, image list')
    parser.add_argument('--mode', default='coarse', choices=['coarse', 'detail'], type=str,
                        help='visualization mode: coarse (fast startup) or detail (with detailed skin support)')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help='set device: mps, cuda, or cpu' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model' )

    main(parser.parse_args())
