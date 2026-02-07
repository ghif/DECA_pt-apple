
import torch
import numpy as np
import cv2
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg

def inspect_shading():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    deca_cfg.model.use_tex = True
    deca = DECA(config=deca_cfg, device=device)
    
    img_path = 'TestSamples/examples/IMG_0392_inputs.jpg'
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image.transpose(2,0,1)[None, ...]).float()/255.
    image = image.to(device)

    with torch.no_grad():
        codedict = deca.encode(image)
        # We need to see internal ops from render.
        # Let's call decode and see what's in opdict if we added it, or just use deca.render
        opdict, visdict = deca.decode(codedict)
    
    # We want to see transformed_normals
    # Let's use deca.render directly to see what's happening
    verts = opdict['verts']
    trans_verts = opdict['trans_verts']
    albedo = opdict.get('albedo', torch.zeros_like(image))
    
    with torch.no_grad():
        ops = deca.render(verts, trans_verts, albedo)
    
    trans_normals = ops['transformed_normals'] # [bz, V, 3]
    print(f"Transformed Normals Mean: {trans_normals.mean(dim=(0,1))}")
    print(f"Transformed Normals Min: {trans_normals.min(dim=1)[0][0]}")
    print(f"Transformed Normals Max: {trans_normals.max(dim=1)[0][0]}")
    
    print(f"Transformed Vertices Z Mean: {trans_verts[0,:,2].mean()}")
    print(f"Transformed Vertices Z Min: {trans_verts[0,:,2].min()}")
    print(f"Transformed Vertices Z Max: {trans_verts[0,:,2].max()}")
    
    # Check pos_mask logic
    z_normals = trans_normals[0, :, 2]
    print(f"Z Normals stats: mean={z_normals.mean():.4f}, min={z_normals.min():.4f}, max={z_normals.max():.4f}")
    
    visible_count = (z_normals < -0.05).sum().item()
    print(f"Visible vertices (z < -0.05): {visible_count} / {len(z_normals)}")
    
    visible_count_pos = (z_normals > 0.05).sum().item()
    print(f"Vertices (z > 0.05): {visible_count_pos} / {len(z_normals)}")

if __name__ == '__main__':
    inspect_shading()
