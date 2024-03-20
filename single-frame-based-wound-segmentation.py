import os
import torch
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image
# from model import WoundSegmentationNetwork
import numpy as np
from matplotlib import pyplot as plt
import json
import open3d as o3d
from collections import Counter
import math
import pymeshfix
from src.models.model import SegNet
import argparse

# This script accepts kinect fusion results folder, Model weights, Frame number and displays segmented 3D Wound Mesh, Wound Area, Perimeter and Volume

REF_W, REF_H = 3024, 4032


def getExtrinsicIntrinsicFromMetadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    extrinsic_list = np.asarray([i['matrix'] for i in metadata_dict['extrinsic']])
    intrinsic = np.asarray(metadata_dict['intrinsic']['matrix']).T
    intrinsic[0][0] = intrinsic[0][0] * 480 / REF_W
    intrinsic[1][1] = intrinsic[1][1] * 640 / REF_H
    intrinsic[0][2] = intrinsic[0][2] * 480 / REF_W
    intrinsic[1][2] = intrinsic[1][2] * 640 / REF_H
    return extrinsic_list, intrinsic


def project3Dto2D(xyz, A):
    fx = A[0][0]
    fy = A[1][1]
    cx = A[0][2]
    cy = A[1][2]
    u = np.round((fy * xyz[1] / xyz[2]) + cy)
    v = np.round(cx - (fx * xyz[0] / xyz[2]))
    return np.asarray([u, v])


def vid2frame(sourcePath, destPath=None):
    if not os.path.exists(destPath):
        os.makedirs(destPath)
        vidObj = cv2.VideoCapture(sourcePath)
        count = 0
        success = 1
        result = []
        while success:
            success, image = vidObj.read()
            if success == 0:
                break
            if count % 1 == 0:
                cv2.imwrite(os.path.join(destPath, '{}.png'.format(count)), image)
            result.append(image)
            count += 1


def extractFrames(args):
    root_path = args.root_path
    vid2frame(
        os.path.join(root_path, 'video.mp4'),
        os.path.join(root_path, 'rgb'))


def nonMaximalSuppression(prob_map):
    baseline_mask = (prob_map > 0.5).astype(np.uint8)
    ret = cv2.connectedComponentsWithStats(baseline_mask, connectivity=8)
    centroid_labels = ret[1]
    print(centroid_labels.shape)
    main_cluster = -1
    max_prob = -1
    for i in range(np.max(centroid_labels) + 1):
        mean_prob = np.mean(prob_map[centroid_labels == i])
        if mean_prob > max_prob:
            max_prob = mean_prob
            main_cluster = i
    baseline_mask[centroid_labels != main_cluster] = 0
    return baseline_mask


def meshPostProcessing(mesh):
    mesh_vertices = mesh.vertices
    pcd = o3d.geometry.PointCloud(mesh_vertices)
    cluster_labels = pcd.cluster_dbscan(eps=3, min_points=5)
    cluster_labels = np.asarray(cluster_labels)
    max_dist = -10000
    main_cluster = -1
    for i in range(np.max(cluster_labels) + 1):
        mean_z = np.mean(np.asarray(mesh_vertices)[cluster_labels == i][:, 2])
        if mean_z > max_dist:
            max_dist = mean_z
            main_cluster = i
    vertex_mask = np.zeros_like(cluster_labels)
    vertex_mask[cluster_labels == main_cluster] = 1
    mesh.remove_vertices_by_mask(1 - vertex_mask)
    o3d.visualization.draw_geometries([mesh])
    # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(cleaned_mesh[0].vertices)])


def getPerimeter(mesh):
    nm_edges_plus_boundary = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    nm_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    boundary_edges = []
    mesh_vertices = np.asarray(mesh.vertices)
    for i in nm_edges_plus_boundary:
        if i not in nm_edges:
            boundary_edges.append(i)
    ret = 0
    for i1, i2 in boundary_edges:
        ret += math.dist(mesh_vertices[i1], mesh_vertices[i2])
    return ret


def extractMasks(args):
    root_path = args.root_path
    mesh_path = os.path.join(root_path, 'truedepth_mesh.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.asarray(mesh.vertex_colors)
    mesh_vertex_normals = np.asarray(mesh.vertex_normals)

    if not os.path.exists(os.path.join(root_path, 'mask')):
        os.makedirs(os.path.join(root_path, 'mask'))
    model = SegNet(
        height=640,
        width=480,
        num_classes=1,
        encoder_rgb='resnet34',
        encoder_depth='resnet34',
        encoder_block='NonBottleneck1D',
        activation='relu',
        encoder_decoder_fusion='add',
        context_module='ppm',
        nr_decoder_blocks=[3] * 3,
        channels_decoder=[512, 256, 128],
        fuse_depth_in_rgb_encoder='SE-add',
        upsampling='learned-3x3-zeropad'
    )
    model.load_state_dict(torch.load(
        args.model_weights_path,
        map_location=torch.device('cpu'))['model'])

    model.eval()
    model.cpu()
    extrinsic_list, intrinsic = getExtrinsicIntrinsicFromMetadata(os.path.join(root_path, 'metadata.json'))
    frame_id = args.frame_id
    vertex_labels = np.zeros(mesh_vertices.shape[0])
    wound2d = []
    wound3d = []
    extrinsic = extrinsic_list[frame_id]
    img = Image.open(os.path.join(root_path, 'rgb/{}. png'.format(frame_id)))
    depth = Image.open(os.path.join(root_path, 'depth/{}. png'.format(frame_id)))
    # img_resized = img.resize((240, 320))
    img_tensor = (torch.from_numpy(np.asarray(img)) / 255).permute((2, 0, 1)).unsqueeze(0)
    depth_tensor = (torch.from_numpy(np.asarray(depth)) / 65535).unsqueeze(0).unsqueeze(0)
    out = model(img_tensor, depth_tensor)
    out_prob = cv2.resize(torch.nn.Sigmoid()(out).squeeze().detach().cpu().numpy(), (480, 640),
                          interpolation=cv2.INTER_LINEAR)
    mask = nonMaximalSuppression(out_prob)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(out_prob)
    plt.show()
    # exit()
    # mask = (out_prob > 0.6).astype(np.uint8)
    cv2.imwrite(os.path.join(root_path, 'mask/{}.png'.format(frame_id)), mask * 255)
    point_pixel_map = (np.zeros((640, 480)) - 1).astype(int)
    vertex_id = 0
    proj_mesh_vertices = []
    extrinsic = np.linalg.inv(extrinsic.T)
    for x, y, z in mesh_vertices:
        hom_xyz = np.ones([4, 1]).astype(np.float32)
        hom_xyz[0] = x
        hom_xyz[1] = y
        hom_xyz[2] = z


        proj_xyz = extrinsic @ hom_xyz
        # print(extrinsic)
        # print(hom_xyz)
        # exit()
        proj_xyz = np.transpose(proj_xyz[:3], axes=[1, 0])
        proj_mesh_vertices.append(proj_xyz[0])
    #
    # proj_mesh_vertices = np.asarray(proj_mesh_vertices)
    # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_vertices)),
    #                                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj_mesh_vertices))])

        gt_uv = project3Dto2D(proj_xyz[0], intrinsic).astype(int)

        if 0 <= gt_uv[0] < 640 and 0 <= gt_uv[1] < 480:
            curr_label = mask[gt_uv[0]][gt_uv[1]]
            if curr_label == 0:
                vertex_labels[vertex_id] = 0
            else:
                # vertex_labels[vertex_id] = 1
                if point_pixel_map[gt_uv[0]][gt_uv[1]] != -1:
                    prev_vertex_id = point_pixel_map[gt_uv[0]][gt_uv[1]]
                    prev_x, prev_y, prev_z = mesh_vertices[prev_vertex_id]
                    dist = math.dist((prev_x, prev_y, prev_z), (x, y, z))
                    # print(dist)
                    if z > prev_z:
                        point_pixel_map[gt_uv[0]][gt_uv[1]] = vertex_id
                        if dist > 5:
                            vertex_labels[prev_vertex_id] = 0
                        vertex_labels[vertex_id] = 1
                    elif z <= prev_z:
                        if dist < 5:
                            vertex_labels[vertex_id] = 1
                else:
                    point_pixel_map[gt_uv[0]][gt_uv[1]] = vertex_id
                    vertex_labels[vertex_id] = 1
        vertex_id += 1
    #
    #     # for i in range(640):
    #     #     for j in range(480):
    #     #         temp_vid = point_pixel_map[i][j]
    #     #         temp_label = mask[i][j]
    #     #         if temp_vid != -1:
    #     #             if temp_label == 1:
    #     #                 vertex_labels[temp_vid] = 1
    #     #                 wound2d.append([i, j])
    #     #                 wound3d.append(mesh_vertices[temp_vid])
    #
    # break
    #
    # frame_id += 10
    # wound2d = np.asarray(wound2d)
    # wound3d = np.asarray(wound3d)
    # print(wound2d.shape)
    # print(wound3d.shape)
    # print(Counter(vertex_labels))

    # mesh.paint_uniform_color([0, 1, 0])

    # vertex_colors = np.asarray(mesh.vertex_colors)
    # vertex_colors[vertex_labels == 1] = [1, 0, 0]
    #
    # mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = mesh.vertices
    # pcd.colors = mesh.vertex_colors
    #
    # o3d.visualization.draw_geometries([pcd])

    # projected_mask = np.zeros_like(mask)
    # for i, j in wound2d:
    #     projected_mask[i][j] = 1
    # plt.imshow(projected_mask)
    # plt.show()

    # hull = np.squeeze(cv2.convexHull(wound2d, clockwise=True, returnPoints=False))

    # wound_face_ids = []
    # for index, (x, y, z) in enumerate(np.asarray(mesh.triangles)):
    #     if vertex_labels[x] == 1 and vertex_labels[y] == 1 and vertex_labels[z] == 1:
    #         wound_face_ids.append(index)
    o3d.visualization.draw_geometries([mesh])
    mesh.remove_vertices_by_mask(1 - vertex_labels)

    # wound_face_ids = np.asarray(wound_face_ids)
    # wound_faces = np.asarray(mesh.triangles)[wound_face_ids]
    # wound_vertex_ids = np.unique(wound_faces)
    # wound_xyz = mesh_vertices[wound_vertex_ids]
    # wound_colors = mesh_colors[wound_vertex_ids]
    # wound_vertex_normals = mesh_vertex_normals[wound_vertex_ids]
    #
    #
    # wound_mesh = o3d.geometry.TriangleMesh()
    # wound_mesh.vertices = o3d.utility.Vector3dVector(wound_xyz)
    # wound_mesh.vertex_colors = o3d.utility.Vector3dVector(wound_colors)
    # wound_mesh.vertex_normals = o3d.utility.Vector3dVector(wound_vertex_normals)
    # wound_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(wound_faces))
    # a, b, c = mesh.cluster_connected_triangles()

    o3d.visualization.draw_geometries([mesh])
    wound_perimeter = getPerimeter(mesh)
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'wound_mesh.ply'), wound_mesh, write_ascii=True)
    meshPostProcessing(mesh)

    meshfix = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

    meshfix.plot()
    meshfix.repair()
    meshfix.plot()
    closed_wound_mesh = o3d.geometry.TriangleMesh()
    closed_wound_mesh.vertices = o3d.utility.Vector3dVector(meshfix.v)
    closed_wound_mesh.triangles = o3d.utility.Vector3iVector(meshfix.f)
    closed_wound_mesh.compute_vertex_normals()
    closed_wound_mesh.compute_triangle_normals()
    o3d.visualization.draw_geometries([closed_wound_mesh])
    wound_area = mesh.get_surface_area()
    wound_volume = closed_wound_mesh.get_volume()
    # wound_convex_hull = mesh.compute_convex_hull()
    # o3d.visualization.draw_geometries([wound_convex_hull[0]])
    print("Wound Volume is {}mm3".format(wound_volume))
    print("Wound Area is {}mm2".format(wound_area))
    print("Wound Perimeter is {}mm".format(wound_perimeter))


def extractMesh(args):
    extractFrames(args)
    extractMasks(args)

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--model_weights_path', type=str, required=True)
parser.add_argument('--frame_id', type=int, required=True)
args = parser.parse_args()

extractMesh(args)