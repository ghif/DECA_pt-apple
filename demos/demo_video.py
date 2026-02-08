# -*- coding: utf-8 -*-
"""
This script generates a video animation of 3D face reconstruction and expression animation from a single image.
"""
import os, sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    device = args.device
    savefolder = args.savefolder
    os.makedirs(savefolder, exist_ok=True)

    # 1. Setup DECA
    deca_cfg.model.use_tex = True
    deca_cfg.rasterizer_type = 'standard'
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)

    # 2. Load Input Image
    testdata = datasets.TestData(args.inputpath, iscrop=True, face_detector='fan', device=device)
    if len(testdata) == 0:
        print(f"No images found in {args.inputpath}")
        return

    # Use the requested image
    image_item = testdata[0]
    images = image_item['image'].to(device)[None,...]
    imagename = image_item['imagename']
    print(f"Loaded image: {imagename}")

    # 3. Initial Reconstruction
    with torch.no_grad():
        codedict = deca.encode(images)
        # Use detail for high quality rendering
        opdict, visdict = deca.decode(codedict, return_vis=True, use_detail=True)

    # 4. Define Animation Sequence
    # Base parameters
    base_exp = codedict['exp'].clone()
    base_pose = codedict['pose'].clone()
    
    # Target Expressions (heuristics from demo_viser.py)
    # Neutral
    exp_neutral = torch.zeros_like(base_exp)
    pose_neutral = base_pose.clone()
    pose_neutral[:, 3:] = 0 # reset jaw

    # Smile
    exp_smile = torch.zeros_like(base_exp)
    exp_smile[:, 0] = 2.0
    exp_smile[:, 1] = 1.0
    pose_smile = pose_neutral.clone()

    # Surprised
    exp_surprised = torch.zeros_like(base_exp)
    exp_surprised[:, 3] = 2.0
    pose_surprised = pose_neutral.clone()
    pose_surprised[:, 3] = 0.5 # Jaw open

    # Keyframes: (exp, pose, frames)
    keyframes = [
        (base_exp, base_pose, 30), # Hold initial for 1s
        (exp_neutral, pose_neutral, 30), # To neutral
        (exp_smile, pose_smile, 45),   # To smile
        (exp_surprised, pose_surprised, 45), # To surprised
        (base_exp, base_pose, 30) # Back to original
    ]

    # Setup Video Writer
    # We will use a 3-panel view: Input | Rendered Geometry (Coarse) | Rendered Geometry (Detail)
    # Each panel is 224x224 (default DECA size)
    fps = 30
    width = 224 * 3
    height = 224
    video_path = os.path.join(savefolder, f"{imagename}_animation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"Generating animation video: {video_path}")
    
    current_exp = keyframes[0][0].clone()
    current_pose = keyframes[0][1].clone()

    for k in range(len(keyframes) - 1):
        start_exp, start_pose, _ = keyframes[k]
        end_exp, end_pose, num_frames = keyframes[k+1]
        
        for i in tqdm(range(num_frames), desc=f"Phase {k}"):
            alpha = i / float(num_frames)
            
            # Interpolate
            interp_exp = (1 - alpha) * start_exp + alpha * end_exp
            interp_pose = (1 - alpha) * start_pose + alpha * end_pose
            
            codedict['exp'] = interp_exp
            codedict['pose'] = interp_pose
            
            with torch.no_grad():
                opdict, visdict = deca.decode(codedict, return_vis=True, use_detail=True)
            
            # Create Frame
            input_img = util.tensor2image(visdict['inputs'][0])
            shape_img = util.tensor2image(visdict['shape_images'][0])
            detail_img = util.tensor2image(visdict['shape_detail_images'][0])
            
            # Concatenate
            frame = np.concatenate([input_img, shape_img, detail_img], axis=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

    out.release()
    print("Video generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Animation Video Generation')
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/ghif_face1.jpg', type=str)
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu', type=str)
    main(parser.parse_args())
