#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from lpipsPyTorch import lpips

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.logger_utils import training_report, prepare_output_and_logger

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# from prune_train import prepare_output_and_logger, training_report
from icecream import ic
from os import makedirs
from prune import prune_list, calculate_v_imp_score
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


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
    从输入点云开始训练 3DGS，并在指定 iteration 插入 LightGaussian 剪枝。

    主流程可以理解成：
        Scene/GaussianModel 初始化
        -> 每轮随机取一个训练相机
        -> render() 得到图像和可见 Gaussian 信息
        -> L1 + DSSIM loss 反传
        -> 早期 densify/prune 调整 Gaussian 数量
        -> 到 args.prune_iterations 时用全局重要性分数剪枝
        -> 保存 PLY/checkpoint/imp_score

    这个脚本适合“从头训练并顺便剪枝”；如果已经有一个训练好的 point_cloud.ply，
    更常用 prune_finetune.py 做加载、剪枝和恢复训练。
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # checkpoint 包含 Gaussian 参数和 optimizer 状态，restore 后可以继续训练。
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 在 3DGS 原始 xyz 学习率调度外，再给所有参数组套一个指数 scheduler。
    # 这里主要服务 LightGaussian 的训练/剪枝恢复实验配置。
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.97)
    for iteration in range(first_iter, opt.iterations + 1):
        # network_gui 是原版 3DGS 的远程预览接口。没有 GUI 连接时这段基本空转；
        # 有连接时可以用自定义相机实时查看当前 Gaussian 渲染结果。
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
        # 每 1000 次迭代增加一次 SH 阶数，让训练从低频颜色逐渐过渡到高阶视角相关外观。
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            gaussians.scheduler.step()

        # Pick a random Camera
        # viewpoint_stack 是一个“无放回”相机池：一轮内尽量遍历所有训练视角，
        # 用完后重新复制并随机采样。这样比每次从全体相机有放回采样更均匀。
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        # render_pkg 中除了图像，还包含 densify 需要的 viewspace_points、visibility_filter、radii。
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
        # 3DGS 的基础 photometric loss：
        #   L1 保证像素颜色接近；
        #   1-SSIM 保持局部结构一致；
        # lambda_dssim 控制两者权重。
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
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
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

            # Densification
            # Densification 是原版 3DGS 的结构自适应阶段。
            # LightGaussian 的剪枝通常发生在 densify 之后或中后期：先让模型长出足够结构，
            # 再根据全局贡献删掉冗余 Gaussian。
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # max_radii2D 记录每个 Gaussian 在屏幕上的最大投影半径。
                # 过大的屏幕投影通常意味着一个 Gaussian 覆盖太广，后续会被 split 或 prune。
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                # add_densification_stats 使用 render() 中屏幕空间点的梯度，
                # 统计每个可见 Gaussian 对当前 loss 的敏感度。
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    # 周期性降低 opacity 可以防止早期点过度占据 alpha，给新点学习机会。
                    gaussians.reset_opacity()

            if iteration in args.prune_iterations:
                # TODO Add prunning types
                # LightGaussian 的关键步骤：遍历所有训练相机，统计每个 Gaussian 的全局重要性。
                # prune_list() 调用 count_render()，后者会走修改过的 CUDA rasterizer，
                # 返回 gaussians_count 和 important_score。
                gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                i = args.prune_iterations.index(iteration)
                # 论文中的 volume-aware score：重要性分数乘上体积修正项。
                # 这样可以减少“像素贡献低但覆盖空间结构较大”的 Gaussian 被过度删除的风险。
                v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                gaussians.prune_gaussians(
                    (args.prune_decay**i) * args.prune_percent, v_list
                )



            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                if iteration == checkpoint_iterations[-1]:
                    # 最后一个 checkpoint 额外保存 imp_score.npz，后续 VecTree 量化会用它决定
                    # 哪些高重要性 Gaussian 保持原始 SH，哪些低重要性 Gaussian 进入 VQ。
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 


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
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument(
        "--prune_iterations", nargs="+", type=int, default=[16_000, 24_000]
    )
    parser.add_argument("--prune_percent", type=float, default=0.5)
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--prune_decay", type=float, default=0.8)
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
