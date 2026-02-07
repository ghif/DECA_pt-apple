
import torch
import numpy as np
import os
import cv2
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets.datasets import TestData

def check_normals():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    deca_cfg.model.use_tex = True
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    
    img_path = 'TestSamples/examples/IMG_0392_inputs.jpg'
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image.transpose(2,0,1)[None, ...]).float()/255.
    image = image.to(device)

    with torch.no_grad():
        codedict = deca.encode(image)
        opdict, visdict = deca.decode(codedict)
    
    normals = opdict['uv_detail_normals']
    coarse_normals = opdict['normals'] # These are [bz, N, 3] or [bz, 3, H, w]? Let's check.
    # Actually in decode: opdict['normals'] = ops['normals'] which is [bz, 3, 256, 256] from forward?
    # No, ops['normals'] is [bz, N, 3] in decoder.
    # We want top-level uv maps.
    
    uv_coarse_normals = deca.render.world2uv(opdict['normals']) # Wait, opdict['normals'] is already in world? 
    # Actually ops['normals'] are vertex normals.
    
    print(f"Detail Normals Mean: {normals.mean(dim=(0,2,3))}")
    
    # Check ops['normals'] stats
    coarse_v_normals = opdict['normals']
    print(f"Coarse Vertex Normals Mean: {coarse_v_normals.mean(dim=(0,1))}")
    
    # Save a visualization of normals Z component for both
    z_normals = normals[0, 2, :, :].cpu().numpy()
    z_vis = (z_normals * 0.5 + 0.5) * 255
    cv2.imwrite('debug_output/detail_normal_z.jpg', z_vis.astype(np.uint8))
    
    # Let's also check the actual shading
    shading = visdict['shape_detail_images'][0].mean().item()
    shape_shading = visdict['shape_images'][0].mean().item()
    print(f"Detail Shading Mean: {shading:.4f}")
    print(f"Shape Shading Mean: {shape_shading:.4f}")

if __name__ == '__main__':
    check_normals()
