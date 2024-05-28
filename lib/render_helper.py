import os
import torch

import cv2

import numpy as np

from PIL import Image
import trimesh
import matplotlib.colors as mcolors

from torchvision import transforms
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
)

# customized
import sys
sys.path.append(".")


def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel,
                                            max_faces_per_bin=30000)  # bin_size=80, 
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer


@torch.no_grad()
def render(mesh, edge_mesh, renderer, pad_value=10, controlnet_cond=None, edge_selector=None, camera_pos=None):
    def canny_edge_detector(img, low=25, high=100, blur=True):
        if blur:
            img = cv2.GaussianBlur(img, (5,5), sigmaX=0, sigmaY=0)
        all_edges = []
        for i in range(3):  # loop over the R, G, and B channels
            edges = cv2.Canny(img[:, :, i], low, high)
            all_edges.append(edges)
        all_edges = np.stack(all_edges, axis=-1)
        all_edges = np.mean(all_edges, axis=-1)
        return all_edges

    def phong_custom_shader(meshes, fragments, vert_feats=None, faces_feats=None) -> torch.Tensor:
        if vert_feats is not None:
            assert faces_feats is None
            assert len(meshes.verts_packed()) == len(vert_feats)
            faces = meshes.faces_packed()  # (F, 3)
            faces_feats = vert_feats[faces]
        else:
            assert faces_feats is not None
            assert len(faces_feats) == len(meshes.faces_packed()) and len(faces_feats.shape) == 3, \
                f"{len(faces_feats)} {len(meshes.faces_packed())} {len(faces_feats.shape)}"

        pixel_feats = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_feats
        )
        return pixel_feats

    # NOTE: Can be optimized
    def get_mesh_edges(meshes, fragments, n_conn_comp_imgs=4, use_cc_edges=True, use_normal_edges=True, use_depth_edges=True):
        # Extract the connected components
        tm_mesh = trimesh.Trimesh(meshes.verts_packed().cpu().numpy(), meshes.faces_packed().cpu().numpy(), process=False)
        conn_components = trimesh.graph.connected_component_labels(tm_mesh.face_adjacency)  # face connected connected-components

        # A list of distinct colors (RGB)
        colors = [mcolors.to_rgb(v) for k, v in mcolors.CSS4_COLORS.items()]  # CSS4_COLORS TABLEAU_COLORS
        colors = (np.array(colors) * 255).astype(np.uint8)

        # =====================================================================================

        # Mask Image
        mask_image = torch.ones_like(fragments.pix_to_face[0])
        mask_image[fragments.pix_to_face[0] == -1] = 0.
        mask_image = mask_image.repeat(1, 1, 3)
        mask_image = (mask_image.cpu().numpy() * 255).astype(np.uint8)

        # =====================================================================================
        
        # Connected Components Image (in random colors)
        conn_comp_imgs = []
        for i in range(n_conn_comp_imgs):
            rand_face_colors = np.random.randint(0, len(colors), size=(len(np.unique(conn_components)), 1))
            rand_face_colors = np.hstack([rand_face_colors] * 3)
            rand_face_colors = colors[rand_face_colors]  # RGB for each vertex of the face of the cc
            rand_face_colors = rand_face_colors[conn_components]
            rand_face_colors = torch.from_numpy(rand_face_colors).to(torch.float32).to(meshes.verts_packed().device)
            conn_comp_image = phong_custom_shader(meshes, fragments, faces_feats=rand_face_colors)

            conn_comp_image = (conn_comp_image[0, ..., 0, :].cpu().numpy() * 255).astype(np.uint8)
            conn_comp_image = cv2.bitwise_and(conn_comp_image, mask_image)
            conn_comp_imgs.append(conn_comp_image)

        # =====================================================================================

        # Using vertex normals
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        vertex_normals = vertex_normals - camera_pos.to(vertex_normals.device)  # To normalize the output color
        faces_normals = vertex_normals[faces]
        # # Using face normals
        # faces_normals = meshes.faces_normals_packed()  # (F, 3)
        # faces_normals = torch.stack([faces_normals] * 3, dim=-1)
        faces_normals = renderer.rasterizer.cameras.transform_points(faces_normals)  # To normalize the output color
        normal_image = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )[0, ..., 0, :]

        normal_image = torch.nn.functional.normalize(normal_image, dim=-1)
        normal_image = (normal_image + 1) / 2  # (0, 1)
        normal_image = (normal_image.cpu().numpy() * 255).astype(np.uint8)
        normal_image = cv2.bitwise_and(normal_image, mask_image)

        # =====================================================================================

        depth_image = fragments.zbuf

        # Post-process the depth map
        depth_n_std = 3
        no_depth = -1  # depth_image is -1 at empty regions
        target_min = 20
        target_max = 255
        no_depth_target_value = 0
        # Clip values to n_std
        depth_mean = torch.mean(depth_image[depth_image != no_depth])
        depth_std = torch.std(depth_image[depth_image != no_depth])
        d_min = torch.maximum(depth_mean - depth_n_std * depth_std, torch.min(depth_image[depth_image != no_depth]))
        d_max = torch.minimum(depth_mean + depth_n_std * depth_std, torch.max(depth_image[depth_image != no_depth]))
        clip_depth_image = torch.clip(depth_image, d_min, d_max)
        # Normalize
        normalize_min = torch.min(clip_depth_image[depth_image != no_depth])
        normalize_max = torch.max(clip_depth_image[depth_image != no_depth])
        clip_depth_image = (clip_depth_image - normalize_min) / (normalize_max - normalize_min)
        # Invert depth (After this: 0->far, 1->close)
        clip_depth_image = 1.0 - clip_depth_image
        # Add a small threshold to diffrentiate object from the background
        clip_depth_image = clip_depth_image * (target_max - target_min) + target_min
        # Set no_depth value
        clip_depth_image[depth_image == no_depth] = no_depth_target_value
        clip_depth_image = torch.cat([clip_depth_image] * 3, dim=-1)
        # return clip_depth_image / 255.0  # (1, render_res, render_res, 3) | [0, 1]
        clip_depth_image = clip_depth_image[0].detach().cpu().numpy().astype(np.uint8)
    
        # =====================================================================================

        # Extract edges
        conn_comp_edge = None
        for img in conn_comp_imgs:
            edge = canny_edge_detector(img, 0, blur=True)
            if conn_comp_edge is None:
                conn_comp_edge = edge
            else:
                conn_comp_edge += edge
        conn_comp_edge = conn_comp_edge.clip(0, 255)

        normal_edge = canny_edge_detector(normal_image, blur=True)

        clip_depth_image = canny_edge_detector(clip_depth_image, blur=False)

        edge_map = np.zeros_like(clip_depth_image)
        if use_cc_edges:
            edge_map += conn_comp_edge
        if use_normal_edges:
            edge_map += normal_edge
        if use_depth_edges:
            edge_map += clip_depth_image
        edge_map = edge_map.clip(0, 255)
        edge_map = torch.from_numpy(edge_map).to(torch.float32).to(meshes.verts_packed().device).unsqueeze(0) # / 255.0

        return edge_map  # (render_res, render_res, 3) | [0, 255]

    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        # Using vertex normals
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        vertex_normals = vertex_normals - camera_pos.to(vertex_normals.device)  # To normalize the output color
        faces_normals = vertex_normals[faces]
        # # Using face normals (not normalized yet)
        # faces_normals = meshes.faces_normals_packed()  # (F, 3)
        # faces_normals = torch.stack([faces_normals] * 3, dim=-1)
        faces_normals = renderer.rasterizer.cameras.transform_points(faces_normals)  # To normalize the output color
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        return pixel_normals

    def similarity_shading(meshes, fragments):
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        vertices = meshes.verts_packed()  # (V, 3)
        face_positions = vertices[faces]
        view_directions = torch.nn.functional.normalize((renderer.shader.cameras.get_camera_center().reshape(1, 1, 3) - face_positions), p=2, dim=2)
        cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals, view_directions)
        pixel_similarity = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, cosine_similarity.unsqueeze(-1)
        )

        return pixel_similarity

    def get_relative_depth_map(fragments, pad_value=pad_value):
        absolute_depth = fragments.zbuf[..., 0] # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black

        return relative_depth


    images, fragments = renderer(mesh)
    normal_maps = phong_normal_shading(mesh, fragments).squeeze(-2)
    similarity_maps = similarity_shading(mesh, fragments).squeeze(-2) # -1 - 1
    
    # depth_maps are depth/edge maps based on the input arguments
    if controlnet_cond == "depth":
        depth_maps = get_relative_depth_map(fragments)
    elif controlnet_cond == "canny":
        fragments_edge_mesh = renderer.rasterizer(edge_mesh)
        depth_maps = get_mesh_edges(edge_mesh, fragments_edge_mesh, n_conn_comp_imgs=4, **edge_selector)
    else:
        raise ValueError()

    # normalize similarity mask to 0 - 1
    similarity_maps = torch.abs(similarity_maps) # 0 - 1
    
    # HACK erode, eliminate isolated dots
    non_zero_similarity = (similarity_maps > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_maps.device).unsqueeze(0) / 255.
    similarity_maps = non_zero_similarity.unsqueeze(-1) * similarity_maps 

    return images, normal_maps, similarity_maps, depth_maps, fragments


@torch.no_grad()
def check_visible_faces(mesh, fragments):
    pix_to_face = fragments.pix_to_face

    # Indices of unique visible faces
    visible_map = pix_to_face.unique()  # (num_visible_faces)

    return visible_map

