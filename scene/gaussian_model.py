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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from icecream import ic
from vectree.utils import load_vqgaussian, write_ply_data


class GaussianModel:
    """
    3D Gaussian Splatting 中所有可训练 Gaussian 的容器。

    每一行代表一个 Gaussian，核心参数如下：
        _xyz:            [N, 3]，3D 中心位置，直接在世界坐标中优化。
        _features_dc:    [N, 1, 3]，SH 的 DC 项，可以理解为基础颜色。
        _features_rest:  [N, K-1, 3]，更高阶 SH 系数，表达随视角变化的外观。
        _opacity:        [N, 1]，优化空间中的 opacity，真实 alpha = sigmoid(_opacity)。
        _scaling:        [N, 3]，优化空间中的尺度，真实尺度 = exp(_scaling)，保证为正。
        _rotation:       [N, 4]，四元数，使用前 normalize 成单位旋转。

    这个类还负责非常关键的“结构变化”：
        densify: 根据屏幕空间梯度复制或拆分 Gaussian；
        prune: 根据 opacity、大小或 LightGaussian 的重要性分数删除 Gaussian。

    因为 Gaussian 个数会变，普通地切 tensor 还不够：AdamW 的动量状态也要同步切片
    或拼接，所以本文件中有不少专门维护 optimizer.state 的代码。
    """

    def setup_functions(self):
        # 将优化空间参数映射到物理/渲染空间的激活函数集中放在这里，便于保存、
        # 加载和替换参数时保持同一套约定。
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # scale + rotation 先组成 3x3 线性变换 L，协方差矩阵为 L @ L^T。
            # rasterizer 只需要对称矩阵的 6 个独立分量，因此 strip_symmetric 会压成
            # [xx, xy, xz, yy, yz, zz] 这样的紧凑格式。
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        # active_sh_degree 是“当前训练/渲染实际启用”的 SH 阶数；
        # max_sh_degree 是模型最多能存多少阶。训练初期从 0 阶开始，逐渐升阶，
        # 可以先学稳定的低频颜色，再放开高频视角相关外观。
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # 这些 tensor 初始化为空，真正的数据来源有三种：
        #   1. create_from_pcd(): 从输入点云初始化；
        #   2. load_ply(): 从已训练 point_cloud.ply 恢复；
        #   3. load_vq(): 从 VecTree/VQ 压缩结果恢复。
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        # densify 依赖每个 Gaussian 的屏幕空间梯度均值：
        #   xyz_gradient_accum 累加梯度范数；
        #   denom 记录被可见视角更新了多少次。
        self.xyz_gradient_accum = torch.empty(0)  # empty or frezze
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        # checkpoint 保存的是“完整训练状态”，不只是点云参数。
        # optimizer.state_dict() 必须一起保存，否则 resume 后 AdamW 的动量会丢失，
        # 训练曲线会和连续训练不一致。
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        # restore 的顺序很重要：
        #   1. 先把参数 tensor 放回对象；
        #   2. 调 training_setup() 按这些 tensor 建好 optimizer param_groups；
        #   3. 再 load_state_dict() 把 AdamW 动量等状态恢复到新 param_groups 上。
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # _scaling 在优化空间可以是任意实数；exp 后才是 rasterizer 使用的正尺度。
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # 四元数优化时可能偏离单位长度，使用前 normalize，避免协方差矩阵畸变。
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # rasterizer 需要完整 SH 系数，DC 和高阶项在内部拆开存储是为了给它们
        # 设置不同学习率：f_rest 的学习率会比 f_dc 小很多。
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        # _opacity 保存 inverse_sigmoid 空间的值；sigmoid 后才是 [0,1] alpha。
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        # 原版 3DGS 的 coarse-to-fine 训练策略：每隔一段迭代增加一个 SH 阶数。
        # 这样可以避免一开始高阶 SH 过度拟合噪声或视角相关细节。
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def onedownSHdegree(self):
        # LightGaussian 的 SH distillation 会把 teacher 的高阶 SH 压到 student 的低阶 SH。
        # 这里把 features_rest 按新的阶数裁短，只保留低阶系数。
        if self.active_sh_degree > self.max_sh_degree:
            self.active_sh_degree -= 1
            num_coeffs_to_keep = (self.active_sh_degree + 1) ** 2 - 1
        ic(num_coeffs_to_keep)
        self._features_rest = self._features_rest.clone().detach()
        self._features_rest = self._features_rest[:,:num_coeffs_to_keep,:]
        self._features_rest.requires_grad = True

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        从 SfM/COLMAP 或 synthetic 随机点云初始化 Gaussian。

        初始化策略的直觉：
            xyz 来自点云；
            DC SH 来自点云 RGB；
            高阶 SH 置零；
            scale 根据最近邻距离估计，让初始 Gaussian 大致覆盖局部空间；
            rotation 设为单位四元数；
            opacity 从 0.1 开始，给 densify/prune 留出调整空间。
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # distCUDA2 返回每个点到近邻的平方距离。用 sqrt(dist2) 作为初始尺度，
        # 可以让 Gaussian 覆盖局部邻域而不是退化为极小点。log 是因为 _scaling
        # 保存在 exp 的逆空间。
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        # 注意这些成员必须是 nn.Parameter，optimizer 才能追踪并更新它们。
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        为当前 Gaussian 参数创建 optimizer 和 densification 统计缓存。

        参数组按语义拆开是 3DGS 的核心工程细节：
            xyz 需要指数衰减学习率，并按 scene extent 做 spatial_lr_scale；
            f_dc 学基础颜色，学习率较高；
            f_rest 学高阶 SH，学习率较低，避免视角相关颜色过早震荡；
            opacity/scale/rotation 分别使用不同学习率。
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        # 只对 xyz 参数使用 3DGS 的指数学习率调度。
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        # PLY 是按列保存的结构化数组；这里定义列名顺序，save_ply/load_ply
        # 必须保持一致，否则会读错属性。
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def construct_list_of_compress_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self.centroids.shape[1]):
            l.append("centroids_{}".format(i))
        for i in range(self.idx.shape[1]):
            l.append("idx_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))

        return l

    def save_ply(self, path):
        # 保存时写“优化空间”的原始参数，而不是 sigmoid/exp 后的值。
        # 这样 load_ply 后继续训练时不会重复做 inverse activation。
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_compress(self, path):
        # 早期压缩格式：把 DC、centroids、idx 和其他几何属性写进 PLY。
        # 当前 VecTree 默认更多使用 extreme_saving/*.npz，但这个函数保留了
        # “压缩表示如何落盘”的历史接口。
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        centroids = self.centroids
        idx = self.idx
        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_compress_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, centroids, idx, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        # densification 期间会周期性把过高 opacity 压回 <=0.01。
        # 这样新拆分/复制出的 Gaussian 有机会重新竞争贡献，避免少数早期点
        # opacity 过大导致结构僵化。
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # Kevin defined function
    def load_ply_sh(self, path, new_sh):
        """
        从完整 PLY 中只加载低阶 SH。

        distill_train.py 会用这个能力把高阶 teacher checkpoint 裁成低阶 student。
        xyz/opacity/scale/rotation 仍完整保留，只有 features_rest 按 new_sh 截断。
        """
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        if new_sh > self.max_sh_degree:
            raise ValueError(
                "Requested max_sh_degree is greater than available in data."
            )
        num_coeffs_to_keep = (new_sh + 1) ** 2
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )
        features_extra = features_extra[:, :, : num_coeffs_to_keep - 1]
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = new_sh
        
        
    def load_vq(self, path):
        # 从 VecTree 量化保存的 extreme_saving 目录还原完整 Gaussian 属性。
        # load_vqgaussian 会把 codebook、vq_index、non_vq_feats、xyz、other_attribute
        # 解包并拼成与 PLY 同顺序的 dense attribute 矩阵。
        # can't load from zip folder
        dequantized_feats = load_vqgaussian(os.path.join(path,'extreme_saving')).cpu().numpy()
        sh_dim = 3*(self.max_sh_degree + 1) ** 2 - 3 
        self.active_sh_degree = self.max_sh_degree
        # ic("in load_vq")
        # 24 for degree 2, and 45 for degree 3
        # abc = dequantized_feats[:, 0:3]
        
        xyz = dequantized_feats[:, 0:3]
        features_dc = dequantized_feats[:, 6:9]
        features_dc = features_dc.reshape((features_dc.shape[0],3,1))
        
        extra_f_names = dequantized_feats[:, 9:9+sh_dim]
        extra_f_names = extra_f_names.reshape((features_dc.shape[0],3,sh_dim//3))
        
        self._xyz = nn.Parameter(
            torch.tensor(dequantized_feats[:, 0:3], dtype=torch.float, device="cuda").requires_grad_(True)
        ) 
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(extra_f_names, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(dequantized_feats[:,-8:-7], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(dequantized_feats[:,-7:-4], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(dequantized_feats[:,-4:], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        
        
    
        

    def load_ply(self, path):
        # 从 point_cloud.ply 读取原始 3DGS/LightGaussian 表示。
        # 属性顺序和 save_ply()/construct_list_of_attributes() 对应：
        # xyz, normals, f_dc, f_rest, opacity, scale, rotation。
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        ic(self.max_sh_degree)
        ic(3 * (self.max_sh_degree + 1) ** 2 - 3)
        # ic(extra_f_names)
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        用一个新 tensor 替换 optimizer 中某个参数组的唯一参数。

        reset_opacity() 会改变 opacity tensor 的数值但不改变 Gaussian 个数。
        这里同时重置 AdamW 的 exp_avg/exp_avg_sq，避免旧动量作用到新 opacity。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        按 valid mask 删除 Gaussian，并同步裁剪 optimizer 状态。

        mask=True 表示保留的点。对于每个参数组：
            参数 tensor 按第一维切片；
            AdamW 的 exp_avg / exp_avg_sq 也按同一 mask 切片；
            再把切好的 tensor 重新包装成 nn.Parameter。

        这是 prune 能继续训练的关键；如果只裁剪 self._xyz 等成员而不处理
        optimizer.state，下一次 optimizer.step() 会因为 shape 不匹配而出错。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # 对外接口使用 mask=True 表示“要删除”，这里取反得到保留点。
        # 删除后所有 per-Gaussian 缓存也要同步裁剪，保持 N 一致。
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        densify 新增 Gaussian 时，把新 tensor 拼接到每个 optimizer 参数组后面。

        新增点没有历史动量，所以 AdamW 状态里为它们拼接 0。
        这保证 densify 之后 optimizer.step() 能无缝处理“旧点 + 新点”。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        # clone/split 产生的新 Gaussian 最终都通过这个函数并入模型。
        # 并入后重置 densification 统计，因为 Gaussian 集合已经变化，旧的
        # xyz_gradient_accum/denom/max_radii2D 不再和当前索引一一对应。
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # split 处理“大而且梯度高”的 Gaussian：
        #   梯度高：说明这个区域还解释不好图像；
        #   尺度大：说明一个 Gaussian 覆盖范围太广，可以拆成多个更小 Gaussian。
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # 在当前 Gaussian 的局部坐标系中按尺度采样 N 个偏移，再乘旋转矩阵转到世界系。
        # 新 Gaussian 的尺度除以 0.8*N，让拆分后总体覆盖更细，不只是复制重叠。
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # clone 处理“小而且梯度高”的 Gaussian。
        # 小 Gaussian 已经局部化，不适合 split 成更小点；直接复制一个同参数点，
        # 后续优化会把副本推向不同位置或属性。
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify(self, max_grad, extent):
        # grads 是每个 Gaussian 在可见视角中的平均屏幕空间梯度。
        # NaN 通常来自 denom=0，即某些点在统计窗口内从未可见，置 0 表示不 densify。
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 训练早期的结构自适应：先 clone/split 增加表达力，再删除低 opacity 或
        # 过大投影/过大世界尺度的异常 Gaussian。
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_opacity(self, percent):
        # 简单按 opacity 百分位剪枝。LightGaussian 更常用 prune_gaussians()
        # 配合 count_render 得到的全局重要性分数。
        sorted_tensor, _ = torch.sort(self.get_opacity, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.get_opacity <= value_nth_percentile).squeeze()

        # big_points_vs = self.max_radii2D > max_screen_size
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score: list):
        # import_score 越小越不重要。percent=0.1 表示删除最低 10% 的 Gaussian。
        # 调用者可以传 opacity、可见次数、important_score，或 LightGaussian 的
        # volume-aware important_score。
        ic(import_score.shape)
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # viewspace_point_tensor.grad 来自 gaussian_renderer.render() 中的 means2D
        # 占位 tensor。这里只取 x/y 两个屏幕方向，因为 densify 判断的是投影位置
        # 对图像 loss 的敏感度。
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
