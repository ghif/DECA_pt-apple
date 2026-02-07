
import cv2
import numpy as np
import torch
from decalib.deca import DECA
from decalib.utils.config import cfg

def debug_render():
    device = 'mps'
    deca = DECA(cfg, device)
    img_path = 'TestSamples/examples/IMG_0392_inputs.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    image = torch.tensor(img.transpose(2,0,1)[None,...]).float()/255.
    
    with torch.no_grad():
        codedict = deca.encode(image.to(device))
        opdict, visdict = deca.decode(codedict)
    
    shape = visdict['shape_images'][0].cpu().numpy().transpose(1,2,0)
    print(f"Shape Max: {np.max(shape)}")
    print(f"Shape Mean: {np.mean(shape)}")
    
    # Save intensified
    cv2.imwrite('debug_shape_int.png', (shape * 255 * 10).clip(0, 255).astype(np.uint8))
    cv2.imwrite('debug_shape.png', (shape * 255).astype(np.uint8))

if __name__ == '__main__':
    debug_render()
