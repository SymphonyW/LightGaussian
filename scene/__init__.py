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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    """
    数据集、相机列表和 GaussianModel 的组织层。

    训练/渲染入口脚本通常只需要：
        scene = Scene(dataset_args, gaussians, ...)
        scene.getTrainCameras()
        scene.getTestCameras()
        scene.save(iteration)

    Scene 的职责不是优化参数，而是把不同数据格式统一成同一种运行时结构：
        COLMAP 数据集 -> CameraInfo / BasicPointCloud
        Blender synthetic -> CameraInfo / 随机初始化点云
        checkpoint/PLY/VQ -> GaussianModel
    """
    gaussians: GaussianModel
    # modified
    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        new_sh=0,
        load_vq=False
    ):
        """b
        :param path: Path to colmap scene main folder.

        :param args: ModelParams，包含 source_path、model_path、images、eval 等配置。
        :param gaussians: 外部创建的 GaussianModel，本构造函数负责把数据加载进去。
        :param load_iteration: None 表示从输入点云初始化；-1 表示加载最大 iteration；
            正整数表示加载指定 iteration 的 point_cloud.ply。
        :param new_sh: 非 0 时只加载低阶 SH，用于 SH distillation。
        :param load_vq: True 时从 extreme_saving 目录加载 VecTree/VQ 压缩表示。
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(args.source_path)
        # 数据读取被委托给 dataset_readers.py。这里通过目录特征识别数据集类型：
        #   sparse/                 -> COLMAP / MipNeRF360 / Tanks&Temples 风格；
        #   transforms_train.json   -> Blender/NeRF synthetic 风格。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # 第一次训练时保存一份 input.ply 和 cameras.json 到 model_path，方便之后
            # 渲染、评估和复现实验。加载已有模型时不需要重复写入这些元数据。
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            # 训练相机和测试相机在所有 resolution_scales 下使用同一个随机顺序。
            # 这样多分辨率训练/评估时，相同索引仍对应同一个视角。
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        # cameras_extent 是 NeRF normalization 的半径，后续用于：
        #   1. 缩放 position learning rate；
        #   2. 判断 Gaussian 是“小点 clone”还是“大点 split”；
        #   3. 删除世界空间中过大的异常 Gaussian。
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # temp comment out
            # CameraInfo 只保存原始图像和相机参数；cameraList_from_camInfos 会真正
            # 创建训练使用的 Camera，包括 CUDA tensor 图像、world_view_transform、
            # projection_matrix、camera_center 等。
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
        if load_vq:
            # 量化后的模型不再从 point_cloud/iteration_x/point_cloud.ply 读取，
            # 而是从 model_path/extreme_saving 下的 npz 文件恢复。
            self.gaussians.load_vq(self.model_path)
            
        elif new_sh != 0 and self.loaded_iter:
            # SH 蒸馏：从完整 checkpoint 中读取几何和低阶 SH，丢弃更高阶 SH。
            self.gaussians.load_ply_sh(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
                new_sh,
            )
        elif self.loaded_iter:
            # 常规渲染/继续训练：加载指定 iteration 的完整 point_cloud.ply。
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            # 从头训练：由输入点云初始化 GaussianModel。
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        # 保存格式沿用 3DGS：model_path/point_cloud/iteration_x/point_cloud.ply。
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
