import os
import torch
import cv2
import numpy as np
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets.datasets import TestData

# def main():
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Config
deca_cfg.model.use_tex = True
deca_cfg.model.extract_tex = True
deca_cfg.rasterizer_type = 'standard'

deca = DECA(config=deca_cfg, device=device)

# Load image
# Trying the path from original script, then falling back
imagepath = 'TestSamples/examples/IMG_0392_inputs.jpg'
if not os.path.exists(imagepath):
    imagepath = 'TestSamples/examples/IMG_0392.jpg'
    if not os.path.exists(imagepath):
            # Just try a common one if specific ones fail
            imagepath = 'TestSamples/examples/results/IMG_0392_inputs/IMG_0392_inputs.jpg' 
            # Or assume user has some image.

if not os.path.exists(imagepath):
    print(f"Warning: Image {imagepath} not found. Trying to find any jpg in TestSamples/examples.")
    import glob
    files = glob.glob('TestSamples/examples/*.jpg')
    if not files:
            # Glob might fail if files are ignored? 
            # Try absolute path
            files = glob.glob(os.path.abspath('TestSamples/examples/*.jpg'))
    
    if files:
        imagepath = files[0]
    else:
        print("Error: No input image found.")
        # return

print(f"Processing: {imagepath}")
testdata = TestData(imagepath, iscrop=True, face_detector='fan', device=device)

i = 0
images = testdata[i]['image'].to(device)[None,...]

with torch.no_grad():
    codedict = deca.encode(images)
    # Canonical decode
    opdict, visdict = deca.decode(codedict)

# Calculate depth if missing
if 'depth_images' not in visdict:
    # Use trans_verts for depth. 
    # Note: if deca.py is reverted, these are aligned verts, not canonical.
    # But this is what we have.
    depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
    visdict['depth_images'] = depth_image

# Save results
os.makedirs('debug_output', exist_ok=True)

# Save 6 panels
panels = {
    '1_input': 'inputs',
    '2_landmarks2d': 'landmarks2d',
    '3_landmarks3d': 'landmarks3d',
    '4_shape': 'shape_images',
    '5_detail': 'shape_detail_images',
    '6_depth': 'depth_images',
    '7_uv_texture_gt': 'uv_texture_gt',
    '8_rendered': 'rendered_images',
}

for name, key in panels.items():
    if key in visdict:
        cv2.imwrite(f'debug_output/{name}.jpg', util.tensor2image(visdict[key][0]))
    else:
        print(f"Warning: {key} not found in visdict.")

# Display FLAME albode for debugging
cv2.imwrite(f"debug_output/albedo.jpg", util.tensor2image(opdict['albedo'][0])) 

# # Normals were already moved to visdict and mapped to [0,1] in deca.py
# if 'uv_detail_normals' in visdict:
#     cv2.imwrite('debug_output/uv_detail_normal.jpg', util.tensor2image(visdict['uv_detail_normals'][0]))
# else:
#     print("Warning: uv_detail_normals not found in visdict.")

print("Saved 8 panel images to debug_output/")

# if __name__ == '__main__':
#     main()