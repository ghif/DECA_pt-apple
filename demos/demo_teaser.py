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
This script generates a teaser GIF for DECA.
It performs the following:
1. Reconstructs a 3D head model from input images.
2. Animates the reconstructed head by:
    a. Varying the head pose (yaw) to show the shape from different angles.
    b. Varying the expression (jaw pose) and transferring expressions from other source images.
3. Saves the resulting animations as a 'teaser.gif' in the output folder.
"""

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
import imageio
from skimage.transform import rescale
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, device=device)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector, device=device)
    # DECA
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.use_tex = True
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []
    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        print(f"Processing {name}...")
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            # Extract texture once to use for all animation frames (so it "sticks")
            uv_texture_gt = opdict['uv_texture_gt']
            h, w = images.shape[2:]
        ### show shape with different views and expressions
        visdict_list = []
        max_yaw = 30
        yaw_list = list(range(0,max_yaw,5)) + list(range(max_yaw,-max_yaw,-5)) + list(range(-max_yaw,0,5))
        for k in yaw_list: #jaw angle from -50 to 50
            ## yaw angle
            euler_pose = torch.randn((1, 3))
            euler_pose[:,1] = k#torch.rand((self.batch_size))*160 - 80
            euler_pose[:,0] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            euler_pose[:,2] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].to(device))) 
            codedict['pose'][:,:3] = global_pose
            codedict['cam'][:,:] = 0.
            codedict['cam'][:,0] = 8
            opdict_view, visdict_view = deca.decode(codedict)   
            # Manually render with the FIXED texture to ensure it follows animation
            render_ops = deca.render(opdict_view['verts'], opdict_view['trans_verts'], uv_texture_gt, h=h, w=w)
            
            frame_visdict = {
                'inputs': images, 
                'shape': visdict['shape_detail_images'],
                'rendered': visdict['rendered_images'],
                'pose': visdict_view['shape_detail_images'],
                'rendered_pose': render_ops['images']
            }         
            visdict_list.append(frame_visdict)

        euler_pose = torch.zeros((1, 3))
        global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].to(device))) 
        codedict['pose'][:,:3] = global_pose
        for (i,k) in enumerate(range(0,31,2)): #jaw angle from -50 to 50        
            # expression: jaw pose
            euler_pose = torch.randn((1, 3))
            euler_pose[:,0] = k#torch.rand((self.batch_size))*160 - 80
            euler_pose[:,1] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            euler_pose[:,2] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            jaw_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].to(device))) 
            codedict['pose'][:,3:] = jaw_pose
            opdict_view, visdict_view = deca.decode(codedict)     
            # Manually render with the FIXED texture
            render_ops = deca.render(opdict_view['verts'], opdict_view['trans_verts'], uv_texture_gt, h=h, w=w)

            visdict_list[i]['exp'] = visdict_view['shape_detail_images']
            visdict_list[i]['rendered_exp'] = render_ops['images']
            count = i

        for (i,k) in enumerate(range(len(expdata))): #jaw angle from -50 to 50        
            # expression: jaw pose
            exp_images = expdata[i]['image'].to(device)[None,...]
            exp_codedict = deca.encode(exp_images)
            # transfer exp code
            codedict['pose'][:,3:] = exp_codedict['pose'][:,3:]
            codedict['exp'] = exp_codedict['exp']
            opdict_exp, exp_visdict = deca.decode(codedict)
            # Manually render with the FIXED texture
            render_ops = deca.render(opdict_exp['verts'], opdict_exp['trans_verts'], uv_texture_gt, h=h, w=w)

            if i+count+1 < len(visdict_list):
                visdict_list[i+count+1]['exp'] = exp_visdict['shape_detail_images']
                visdict_list[i+count+1]['rendered_exp'] = render_ops['images']

        visdict_list_list.append(visdict_list)
    
    ### write gif
    writer = imageio.get_writer(os.path.join(savefolder, 'teaser.gif'), mode='I')
    for i in range(len(yaw_list)):
        grid_image_list = []
        for j in range(len(testdata)):
            grid_image = deca.visualize(visdict_list_list[j][i])
            grid_image_list.append(grid_image)
        grid_image_all = np.concatenate(grid_image_list, 0)
        grid_image_all = rescale(grid_image_all, 0.6, channel_axis=-1) # resize for showing in github
        # print(f"grid_image_all shape: {grid_image_all.shape}, dtype: {grid_image_all.dtype}")
        if grid_image_all.max() <= 1.0:
            grid_image_all = (grid_image_all * 255).astype(np.uint8)
        else:
            grid_image_all = grid_image_all.astype(np.uint8)
        writer.append_data(grid_image_all[:,:,[2,1,0]])

    print(f'-- please check the teaser figure in {savefolder}')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/teaser', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/teaser/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    
    if torch.cuda.is_available():
        default_device = 'cuda'
    elif torch.backends.mps.is_available():
        default_device = 'mps'
    else:
        default_device = 'cpu'

    parser.add_argument('--device', default=default_device, type=str,
                        help='set device, cpu for using cpu, mps for macOS' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )

    main(parser.parse_args())