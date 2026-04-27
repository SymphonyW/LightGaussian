#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from icecream import ic
import random
import copy
import gc
import numpy as np
from collections import defaultdict

# from cuml.cluster import HDBSCAN


# def HDBSCAN_prune(gaussians, score_list, prune_percent):
#     # Ensure the tensor is on the GPU and detached from the graph
#     s, d = gaussians.get_xyz.shape
#     X_gpu = cp.asarray(gaussians.get_xyz.detach().cuda())

#     scores_gpu = cp.asarray(score_list.detach().cuda())
#     hdbscan = HDBSCAN(min_cluster_size = 100)
#     cluster_labels = hdbscan.fit_predict(X_gpu)
#     points_by_centroid = {}
#     ic("cluster_labels")
#     ic(cluster_labels.shape)
#     ic(cluster_labels)
#     for i, label in enumerate(cluster_labels):
#         if label not in points_by_centroid:
#             points_by_centroid[label] = []
#         points_by_centroid[label].append(i)
#     points_to_prune = []

#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Skip noise points with label -1
#         if centroid_idx == -1:
#             continue
#         num_to_prune = int(cp.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         point_indices_cp = cp.array(point_indices)
#         distances = scores_gpu[point_indices_cp].squeeze()
#         indices_to_prune = point_indices_cp[cp.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     # points_to_prune now contains the indices of the points to be pruned
#     return mask


# def uniform_prune(gaussians, k, score_list, prune_percent, sample = "k_mean"):
#     # get the farthest_point
#     D, I = None, None
#     s, d = gaussians.get_xyz.shape

#     if sample == "k_mean":
#         ic("k_mean")
#         n_iter = 200
#         verbose = False
#         kmeans = faiss.Kmeans(d, k=k, niter=n_iter, verbose=verbose, gpu=True)
#         kmeans.train(gaussians.get_xyz.detach().cpu().numpy())
#         # The cluster centroids can be accessed as follows
#         centroids = kmeans.centroids
#         D, I = kmeans.index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     else:
#         point_idx = farthest_point_sampler(torch.unsqueeze(gaussians.get_xyz, 0), k)
#         centroids = gaussians.get_xyz[point_idx,: ]
#         centroids = centroids.squeeze(0)
#         index = faiss.IndexFlatL2(d)
#         index.add(centroids.detach().cpu().numpy())
#         D, I = index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     points_to_prune = []
#     points_by_centroid = defaultdict(list)
#     for point_idx, centroid_idx in enumerate(I.flatten()):
#         points_by_centroid[centroid_idx.item()].append(point_idx)
#     for centroid_idx in points_by_centroid:
#         points_by_centroid[centroid_idx] = np.array(points_by_centroid[centroid_idx])
#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Find the number of points to prune
#         num_to_prune = int(np.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         distances = score_list[point_indices].squeeze().cpu().detach().numpy()
#         indices_to_prune = point_indices[np.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     # Convert the list to an array
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     return mask

"""
计算全局显著性分数，基于体积和重要性分数的乘积。
体积是通过高斯组件的缩放参数计算的。然后将体积与第90百分位数的体积进行比较，并将结果提升到v_pow次幂。
最后，将调整后的体积与重要性分数相乘，得到最终的显著性分数列表（v_list）。
这个列表可以用于后续的剪枝步骤，以确定哪些高斯组件应该被保留或移除。
"""
def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.

    计算 LightGaussian 的 volume-aware 重要性分数。

    imp_list 来自 count_render() 的全局累计分数，主要反映一个 Gaussian 在所有训练视角
    的像素混合贡献。但只看 imp_list 容易偏向“经常被看到、局部颜色贡献高”的小 Gaussian，
    对覆盖空间较大的 Gaussian 不够友好。

    因此这里额外乘一个体积项：
        volume = scale_x * scale_y * scale_z
        volume_weight = (volume / 第 90% 位置的体积) ** v_pow
        v_list = volume_weight * imp_list

    v_pow 越大，体积对剪枝排名的影响越强；v_pow=0 时退化为纯 imp_list。
    """
    # Calculate the volume of each Gaussian component
    # 计算每个高斯的体积，体积是通过高斯的缩放参数计算的。
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    # 对体积进行排序，找到第90百分位数的体积值。
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    # 计算v_list，通过将体积与第90百分位数的体积进行比较，并将结果提升到v_pow次幂。
    # 最后，将调整后的体积与重要性分数相乘，得到最终的显著性分数列表（v_list）。
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list



"""
遍历所有的训练视点，使用count_render函数计算每个高斯组件的显著性分数，
并将这些分数累积到gaussian_list和imp_list中。最终返回这两个列表。
gaussian_list表示每个Gaussian被每个视点看到的次数，imp_list表示每个Gaussian的重要性分数的累积。
这些列表可以用于后续的剪枝步骤，以确定哪些高斯组件应该被保留或移除。
"""
def prune_list(gaussians, scene, pipe, background):
    """
    遍历所有训练相机，累计每个 Gaussian 的全局可见/贡献统计。

    返回：
        gaussian_list: 每个 Gaussian 在所有视角中参与渲染的次数统计。
        imp_list: 每个 Gaussian 在所有视角中的重要性分数累计。

    注意：
        这里调用的是 gaussian_renderer.count_render()，不是普通 render()。
        count_render() 依赖项目中修改过的 CUDA rasterizer；原版 3DGS rasterizer
        不会返回 gaussians_count 和 important_score。

    这些统计只用于排序剪枝，不参与反向传播，因此循环里会 detach()。
    """
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    # count_render函数计算viewpoint_cam视点下每个高斯的显著性分数，并将这些分数累积到gaussian_list和imp_list中。
    # 相当于初始化了gaussian_list和imp_list。
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )
    """
    render_pkg包含了每个高斯组件的显著性分数，这些分数被累积到gaussian_list和imp_list中。
    其中，render_pkg["gaussians_count"]表示每个Gaussian被每个视点看到的次数，
    render_pkg["important_score"]表示每个Gaussian的重要性分数的累积。
    """

    # ic(dataset.model_path)
    # 遍历所有的训练视点，使用count_render函数计算每个高斯的显著性分数，并将这些分数累积到gaussian_list和imp_list中。
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        # 这里不随机抽样，而是把训练视角全部跑一遍；这样得到的是“全局”重要性，
        # 不会因为某个 batch 的视角分布偶然性而误剪。
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        # detach()方法用于将张量从计算图中分离出来，使其不再参与梯度计算。这对于在循环中累积显著性分数非常重要，因为我们不希望这些分数影响模型的训练过程。

        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
    return gaussian_list, imp_list
