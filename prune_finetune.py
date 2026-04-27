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
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from icecream import ic
import random
import copy
import gc
from os import makedirs
from prune import prune_list, calculate_v_imp_score
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
from utils.logger_utils import training_report, prepare_output_and_logger


to_tensor = (
    lambda x: x.to("cuda")
    if isinstance(x, torch.Tensor)
    else torch.Tensor(x).to("cuda")
)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(to_tensor([10.0]))


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    """
    加载已有 Gaussian 模型，执行一次或多次剪枝，然后继续 fine-tune 恢复质量。

    和 train_densify_prune.py 的区别：
        train_densify_prune.py: 从输入点云开始训练，并在训练中插入剪枝；
        prune_finetune.py: 从 checkpoint 或 point_cloud.ply 开始，面向已训练模型做压缩。

    典型用法是：
        1. 先用原版/本项目训练得到一个高质量 point_cloud.ply；
        2. 用 --start_pointcloud 或 --start_checkpoint 加载；
        3. 在 args.prune_iterations 指定的位置剪掉低分 Gaussian；
        4. 后续迭代继续优化剩余 Gaussian，恢复 PSNR/SSIM/LPIPS。
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if checkpoint:
        # checkpoint 路径会恢复 optimizer 状态，适合从中断训练继续。
        gaussians.training_setup(opt)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif args.start_pointcloud:
        # point_cloud.ply 只保存 Gaussian 参数，不保存 optimizer 状态；
        # 因此 load_ply 后必须重新 training_setup()。
        gaussians.load_ply(args.start_pointcloud)
        ic(gaussians.get_xyz.shape)
        # ic(gaussians.optimizer.param_groups["xyz"].shape)
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        
    else:
        raise ValueError("A checkpoint file or a pointcloud is required to proceed.")

        
        

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    for iteration in range(first_iter, opt.iterations + 1):
        # 可选 GUI 预览逻辑，和原版 3DGS 训练脚本保持一致。
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # fine-tune 阶段仍保持 SH 逐步升阶策略；如果加载的模型已经是满阶，
        # oneupSHdegree() 不会超过 max_sh_degree。
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 400 == 0:
            gaussians.scheduler.step()

        # Pick a random Camera
        # 随机取一个训练视角做常规 photometric fine-tune。
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        # 剪枝后剩余 Gaussian 的几何/颜色都需要重新适配，所以这里仍用真实图像监督。
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1000 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1000)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                
                if iteration == checkpoint_iterations[-1]:
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 


            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

            if iteration in args.prune_iterations:
                ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
                i = args.prune_iterations.index(iteration)
                # 剪枝前先完整遍历训练视角，得到当前模型的全局统计。
                # 这些分数会随 fine-tune 变化，因此每个 prune iteration 都重新统计。
                gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)

                if args.prune_type == "important_score":
                    # 直接按 rasterizer 统计的像素贡献剪枝。
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, imp_list
                    )
                elif args.prune_type == "v_important_score":
                    # normalize scale
                    # LightGaussian 默认更推荐的分数：像素贡献 * 体积修正。
                    # 它比纯 important_score 更能保留覆盖大空间结构的 Gaussian。
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "max_v_important_score":
                    # 用最大轴尺度做几何权重，和体积权重相比更偏向保留长条/扁平结构。
                    v_list = imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, v_list
                    )
                elif args.prune_type == "count":
                    # 只按可见/参与次数剪枝。这个分数更粗糙，不直接衡量颜色贡献。
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent, gaussian_list
                    )
                elif args.prune_type == "opacity":
                    # 只按 opacity 剪枝，适合作为 baseline，但可能误删低 alpha 细节。
                    gaussians.prune_gaussians(
                        (args.prune_decay**i) * args.prune_percent,
                        gaussians.get_opacity.detach(),
                    )
                # TODO(release different pruning method)
                # elif args.prune_type == "HDBSCAN":
                #     masks = HDBSCAN_prune(gaussians, imp_list, (args.prune_decay**i)*args.prune_percent)
                #     gaussians.prune_points(masks)
                # # elif args.prune_type == "v_important_score":
                # #     imp_list *
                # elif args.prune_type == "two_step":
                #     if i == 0:
                #         volume = torch.prod(gaussians.get_scaling, dim = 1)
                #         index = int(len(volume) * 0.9)
                #         sorted_volume, sorted_indices = torch.sort(volume, descending=True, dim=0)
                #         kth_percent_largest = sorted_volume[index]
                #         v_list = torch.pow(volume/kth_percent_largest, args.v_pow)
                #         v_list = v_list * imp_list
                #         gaussians.prune_gaussians((args.prune_decay**i)*args.prune_percent, v_list)
                #     else:
                #         k = 5^(1*i) * 100
                #         masks = uniform_prune(gaussians, k, imp_list, 0.3, "k_mean")
                #         gaussians.prune_points(masks)
                # else:
                #     k = len(gaussians.get_xyz)//500 * i
                #     masks = uniform_prune(gaussians, k, imp_list, (args.prune_decay**i)*args.prune_percent, args.prune_type)
                #     gaussians.prune_points(masks)
                # gaussians.prune_gaussians(args.prune_percent, imp_list)
                # gaussians.optimizer.zero_grad(set_to_none = True) #hachy way to maintain grad
                # if (iteration in args.opacity_prune_iterations):
                #         gaussians.prune_opacity(0.05)
                else:
                    raise Exception("Unsupportive pruning method")

                ic("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))

            # if iteration in args.densify_iteration:
            # 下面这段保留了“剪枝后再 densify”的实验入口，默认关闭。
            # 开启后会重新增加 Gaussian，压缩率会下降，但有时能恢复被误剪的细节。
            #     gaussians.max_radii2D[visibility_filter] = torch.max(
            #         gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            #     )
            #     gaussians.add_densification_stats(
            #         viewspace_point_tensor, visibility_filter
            #     )
            #     gaussians.densify(opt.densify_grad_threshold, scene.cameras_extent)
            
                ic("after")
                ic(gaussians.get_xyz.shape)
                ic(len(gaussians.optimizer.param_groups[0]['params'][0]))

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[30_001, 30_002, 35_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[35_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[35_000]
    )

    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[30_001])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_pointcloud", type=str, default=None)
    parser.add_argument("--prune_percent", type=float, default=0.1)
    parser.add_argument("--prune_decay", type=float, default=1)
    parser.add_argument(
        "--prune_type", type=str, default="important_score"
    )  # k_mean, farther_point_sample, important_score
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--densify_iteration", nargs="+", type=int, default=[-1])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nTraining complete.")
