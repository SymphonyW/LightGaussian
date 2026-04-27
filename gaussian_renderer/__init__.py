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

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


# 这个文件是 Python 训练代码和 CUDA rasterizer 之间最重要的一层封装：
#
#   Camera + GaussianModel
#          |
#          v
#   render()/count_render()
#          |
#          v
#   diff_gaussian_rasterization.GaussianRasterizer
#
# render() 只关心正常训练/测试需要的渲染图像和可见性信息；
# count_render() 打开 f_count=True，额外让修改过的 rasterizer 返回每个
# Gaussian 的参与次数和重要性分数，这是 LightGaussian 做全局剪枝的关键。


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    渲染单个相机视角。

    输入：
        viewpoint_camera: scene.cameras.Camera，里面已经预计算好了 view/proj 矩阵。
        pc: GaussianModel，保存所有 3D Gaussian 的位置、SH、opacity、scale、rotation。
        pipe: PipelineParams，控制协方差和 SH->RGB 是否在 Python 侧预计算。
        bg_color: CUDA tensor，背景色必须在 GPU 上，因为 rasterizer 在 CUDA 侧使用它。

    输出字典：
        render: 渲染出的 RGB 图像。
        viewspace_points: 2D 屏幕空间均值的占位 tensor，训练时用它拿到屏幕空间梯度。
        visibility_filter: radii > 0 的布尔 mask，表示本视角实际参与渲染的 Gaussian。
        radii: 每个 Gaussian 投影到屏幕后的半径，用于 densification/pruning 判断。
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 这里创建一个和 3D 坐标同形状的 0 tensor，作为 rasterizer 的 means2D 输入。
    # 它不是为了提供真实 2D 坐标，而是为了让自定义 CUDA autograd 函数把
    # “屏幕空间投影位置”的梯度回传到 PyTorch。后面的 densify 逻辑会读取
    # viewspace_point_tensor.grad，并用这个梯度判断哪些 Gaussian 对图像误差敏感。
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # FoV 在 Camera 中以弧度保存；rasterizer 需要 tan(fov/2) 来完成投影和 tile 覆盖计算。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # GaussianRasterizationSettings 是传给 C++/CUDA rasterizer 的完整渲染上下文。
    # 注意 f_count=False：普通 render 只返回图像和 radii，不返回全局重要性统计。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        f_count=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # GaussianModel 内部保存的是“优化空间”的原始参数：
    #   _opacity 经 sigmoid 后才是真实 alpha；
    #   _scaling 经 exp 后才是正尺度；
    #   _rotation 经 normalize 后才是单位四元数。
    # 这里统一通过 property 取值，避免把未激活的参数直接传给 CUDA。
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 协方差有两条路径：
    #   1. pipe.compute_cov3D_python=True：Python 先由 scale+rotation 算出 3D 协方差；
    #   2. False：只传 scale/rotation，CUDA rasterizer 内部再计算。
    # 默认走 CUDA 路径，通常更快；Python 路径方便调试或对照。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 颜色也有三条路径：
    #   override_color != None：外部直接指定 RGB，常用于调试或可视化。
    #   convert_SHs_python=True：Python 根据视线方向把 SH 系数解码成 RGB。
    #   默认：把 SH 系数传给 rasterizer，让 CUDA 在每个 Gaussian 上计算颜色。
    # LightGaussian 的 distillation 和 VecTree 主要压缩的是 SH 特征，所以这里是理解
    # “外观参数如何变成颜色”的关键入口。
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # pc.get_features: [N, SH_coeff_count, 3]
            # eval_sh 期望 [N, 3, SH_coeff_count]，所以先 transpose 再 reshape。
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            # SH 颜色是 view-dependent 的，因此要用 “Gaussian 中心 -> 相机中心”
            # 的方向作为球谐函数的查询方向。
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 真正的 splatting 发生在这里。CUDA 侧会完成：
    #   1. 视锥裁剪；
    #   2. 3D Gaussian 投影到屏幕空间 2D 椭圆；
    #   3. 按 tile 分桶并按深度混合；
    #   4. autograd 反向传播所需的中间量保存。
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # radii<=0 表示该 Gaussian 被视锥裁剪掉，或投影后没有有效屏幕覆盖。
    # 这些点不会贡献图像误差，因此 densification 统计时也不应更新它们。
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def count_render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    渲染并统计每个 Gaussian 的全局重要性。

    这个函数和 render() 的输入基本一致，但 raster_settings.f_count=True。
    修改过的 compress-diff-gaussian-rasterization 会在 CUDA 前向过程中额外累计：
        gaussians_count: 每个 Gaussian 在该视角中实际参与混合/覆盖的计数。
        important_score: 每个 Gaussian 对最终像素混合的贡献强度。

    prune.prune_list() 会遍历所有训练相机，把这些 per-view 分数累加成全局分数。
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 和 render() 一样保留屏幕空间梯度占位。即使剪枝统计主要用 count/score，
    # 返回相同结构也能让上层逻辑复用 render() 的字段。
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # f_count=True 是这个函数和普通 render() 的核心差异。
    # CUDA rasterizer 会走带统计的分支，并把每个 Gaussian 的贡献分数带回 Python。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        f_count=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 下方参数准备逻辑与 render() 保持一致，保证“统计分数”和“实际渲染”
    # 使用同一套几何、协方差和颜色路径。
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # f_count=True 时 rasterizer 的返回值比普通 render 多两个 tensor。
    # 这依赖本项目 submodules 中的定制 rasterizer，不是原版 3DGS rasterizer 的接口。
    gaussians_count, important_score, rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # important_score 后续通常还会和体积项结合，避免只按像素贡献剪掉大但稀疏的结构。
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "gaussians_count": gaussians_count,
        "important_score": important_score,
    }
