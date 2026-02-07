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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import util
from .util import load_obj

# Global variable for the rasterizer function
standard_rasterize = None

def set_rasterizer(type='standard'):
    """
    Initializes the rasterizer. 
    Now defaults to and enforces the 'standard' rasterizer which uses a C++ CPU extension.
    This ensures compatibility with Apple Silicon (MPS) and removes PyTorch3D dependencies.
    """
    global standard_rasterize
    
    print(f"[begin: set_rasterizer] Initializing rasterizer (requested: {type}, using: standard).")
    
    # Compile/Load the Standard Rasterizer (CPU version)
    # Ref: https://pytorch.org/tutorials/advanced/cpp_extension.html
    curr_dir = os.path.dirname(__file__)
    source_path = os.path.join(curr_dir, 'rasterizer', 'standard_rasterize_cpu.cpp')
    
    if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

    print(f"[set_rasterizer] Compiling/Loading standard rasterizer CPU extension...")
    
    try:
        from torch.utils.cpp_extension import load
        standard_rasterize_cpu_module = load(
            name='standard_rasterize_cpu', 
            sources=[source_path],
            verbose=True
        )
        _standard_rasterize_cpu_impl = standard_rasterize_cpu_module.standard_rasterize
        print(f"[set_rasterizer] Successfully loaded standard rasterizer extension.")
    except Exception as e:
        print(f"[set_rasterizer] Error loading C++ extension: {e}")
        raise

    def standard_rasterize_wrapper(faces_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w):
        """
        Wrapper for the C++ standard rasterizer.
        Handles device transfer for MPS/CUDA compatibility by offloading to CPU.
        """
        original_device = faces_vertices.device
        
        # If on MPS or CUDA, move to CPU for rasterization
        if original_device.type != 'cpu':
            faces_vertices_cpu = faces_vertices.cpu()
            depth_buffer_cpu = depth_buffer.cpu()
            triangle_buffer_cpu = triangle_buffer.cpu()
            baryw_buffer_cpu = baryw_buffer.cpu()
            
            _standard_rasterize_cpu_impl(faces_vertices_cpu, depth_buffer_cpu, triangle_buffer_cpu, baryw_buffer_cpu, h, w)
            
            # Move results back to original device
            depth_buffer.copy_(depth_buffer_cpu.to(original_device))
            triangle_buffer.copy_(triangle_buffer_cpu.to(original_device))
            baryw_buffer.copy_(baryw_buffer_cpu.to(original_device))
        else:
            _standard_rasterize_cpu_impl(faces_vertices, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    standard_rasterize = standard_rasterize_wrapper
    print(f"[end: set_rasterizer] Rasterizer initialized.")

class StandardRasterizer(nn.Module):
    """ 
    Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """
    def __init__(self, height, width=None):
        super().__init__()
        if width is None:
            width = height
        self.h = height
        self.w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        device = vertices.device
        if h is None:
            h = self.h
        if w is None:
            w = self.w # Use self.w, fixed bug in original code where it used self.h
            
        bz = vertices.shape[0]
        depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
        triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
        baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
        
        # Clone vertices to avoid modifying original
        vertices = vertices.clone().float()
        # Standard NDC: Y=-1 is Top (Pixel 0), Y=1 is Bottom (Pixel h)
        vertices[..., 0] = vertices[..., 0] * w / 2 + w / 2
        vertices[..., 1] = vertices[..., 1] * h / 2 + h / 2
        # Scale Z similarly if needed by C++ (original code scaled Z by w/2)
        vertices[..., 2] = vertices[..., 2] * w / 2
        
        f_vs = util.face_vertices(vertices, faces)
        
                # Call the wrapper (which handles CPU offloading)
        if standard_rasterize is None:
             raise RuntimeError("Rasterizer not initialized. Please call set_rasterizer() first.")
             
        standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)
        
        pix_to_face = triangle_buffer[:,:,:,None].long()
        bary_coords = baryw_buffer[:,:,:,None,:]
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals

class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256, rasterizer_type='standard'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        
        # Enforce StandardRasterizer
        self.rasterizer = StandardRasterizer(image_size)
        self.uv_rasterizer = StandardRasterizer(uv_size)
        
        # Load OBJ using util.load_obj
        verts, uvcoords, faces, uvfaces = load_obj(obj_filename)
        verts = verts[None, ...]
        uvcoords = uvcoords[None, ...]
        faces = faces[None, ...]
        uvfaces = uvfaces[None, ...]

        # faces
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        # faces
        dense_triangles = util.generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1
        # Flip Y so Forehead (high origin V) becomes NDC Y=-1 (Top)
        uvcoords[..., 1] = -uvcoords[..., 1]
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        
        face_uvcoords = util.face_vertices(self.uvcoords, self.uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([112, 112, 112])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)
    
    def forward(self, vertices, transformed_vertices, albedos, lights=None, h=None, w=None, light_type='point', background=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        # Move mesh so minz larger than 0 for rasterizer (near 0, far 100)
        # Avoid in-place modification to prevent shifts in subsequent calls
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), 
                                transformed_face_normals.detach(), 
                                face_vertices.detach(), 
                                face_normals], 
                                -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] > 0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        if background is not None:
            images = images*alpha_images + background*(1.-alpha_images)
            albedo_images = albedo_images*alpha_images + background*(1.-alpha_images)
        else:
            images = images*alpha_images 
            albedo_images = albedo_images*alpha_images 

        outputs = {
            'images': images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, colors = None, images=None, detail_normal_images=None, 
                lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                [
                [0,0,1]
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()*2.0
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        # Avoid in-place modification
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        if colors is None:
            colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors, 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals,
                        self.face_uvcoords.expand(batch_size, -1, -1, -1)], 
                        -1)
        # rasterize
        # import ipdb; ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        if lights is not None and lights.shape[1] == 9:
            shading_images = self.add_SHlight(normal_images, lights)
        else:
            shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
            shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]; 
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid, alpha_images
        else:
            return shape_images
    
    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        # Avoid in-place modification
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        z = -transformed_vertices[:,:,2:].repeat(1,1,3).clone()
        z = z-z.min()
        z = z/z.max()
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images
    
    def render_colors(self, transformed_vertices, colors):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = colors.shape[0]

        # Attributes
        attributes = util.face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :]* alpha_images
        return images

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        return uv_vertices
