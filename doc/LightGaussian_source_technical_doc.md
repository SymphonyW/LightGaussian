# LightGaussian 源码级技术文档

# 1. 项目整体概述

LightGaussian 解决的是原始 3D Gaussian Splatting 模型体积大、Gaussian 数量多、无界场景渲染负担重的问题。它不是重写 3DGS，而是在原始 3DGS 的 Python 训练框架、Gaussian 参数表示、CUDA rasterizer 和 PLY 存储格式之上，增加了一条压缩链路：

```text
训练好的 3DGS
-> 全局重要性统计
-> Gaussian 剪枝
-> fine-tune 恢复质量
-> SH 降阶蒸馏
-> VecTree / VQ 量化
-> 更小的 Gaussian 表示
```

核心思想从工程角度看是：

* 保留原始 3DGS 的 Gaussian 表示和 tile-based CUDA splatting。
* 在 rasterizer 内额外统计每个 Gaussian 对训练视角像素混合的参与程度。
* 用统计出来的全局重要性分数删除低价值 Gaussian。
* 剪枝后继续优化，恢复 PSNR/SSIM/LPIPS。
* 用 teacher-student 渲染蒸馏把 SH 阶数降下来。
* 用重要性感知 VQ 对 SH 特征做压缩。

和原始 3DGS 的关系：

* `scene/gaussian_model.py`、`gaussian_renderer/__init__.py`、`scene/`、`utils/` 基本继承原始 3DGS 的工程组织。
* `submodules/compress-diff-gaussian-rasterization/` 是修改过的 rasterizer，不是原版 `diff-gaussian-rasterization`。
* LightGaussian 的关键增量在 `prune.py`、`prune_finetune.py`、`distill_train.py`、`vectree/`，以及 CUDA 里的 `count_gaussians` 分支。

# 2. 技术架构设计（重点）

整体架构关系：

```text
数据集
  COLMAP: sparse/0 + images
  Blender: transforms_train.json / transforms_test.json
        |
        v
scene.dataset_readers
  readColmapSceneInfo()
  readNerfSyntheticInfo()
        |
        v
Scene
  Camera list
  GaussianModel
        |
        v
训练 / 压缩入口
  train_densify_prune.py
  prune_finetune.py
  distill_train.py
  vectree/vectree.py
        |
        v
gaussian_renderer
  render()
  count_render()
        |
        v
compress-diff-gaussian-rasterization
  Python autograd wrapper
  C++ binding
  CUDA kernels
        |
        v
输出
  point_cloud.ply
  chkpnt*.pth
  imp_score.npz
  extreme_saving/
  render png frames
```

模块职责：

* 数据处理：`convert.py` 调用 COLMAP；`scene/dataset_readers.py` 读取相机和点云；`utils/camera_utils.py` 构造训练用 Camera。
* Gaussian 表示：`scene/gaussian_model.py` 管理中心、SH、opacity、scale、rotation、optimizer state、PLY/VQ 加载保存。
* 渲染：`gaussian_renderer/__init__.py` 封装 Python API；`submodules/compress-diff-gaussian-rasterization/` 执行 CUDA splatting 和反传。
* 训练：`train_densify_prune.py` 从头训练并剪枝；`prune_finetune.py` 加载已有模型后剪枝恢复；`distill_train.py` 做 SH 蒸馏。
* 剪枝：`prune.py` 统计全局重要性并构造 volume-aware 分数。
* 压缩：`vectree/vectree.py` 和 `vectree/vq.py` 做重要性感知向量量化。
* 渲染输出：`render.py` 渲染 train/test；`render_video.py` 生成轨迹视频帧。

数据流：

```text
输入图像 + 相机参数 + 初始点云
-> Scene 构造 Camera 列表和 GaussianModel
-> training() 随机采样训练相机
-> render() 调 CUDA rasterizer 得到图像
-> L1 + DSSIM loss 反传
-> GaussianModel 根据梯度 densify / prune
-> prune_list() 遍历训练视角统计重要性
-> prune_gaussians() 删除低分 Gaussian
-> fine-tune 恢复质量
-> save_ply() / checkpoint / imp_score.npz
-> distill_train.py 降 SH
-> vectree.py 量化 SH
-> render.py / render_video.py 输出图像
```

# 3. 核心模块深度解析（最重要）

## 数据读取与 Scene 组织

### `scene/dataset_readers.py::readColmapSceneInfo(path, images, eval, llffhold=8)`

作用：读取 COLMAP 数据集，并转换成统一的 `SceneInfo`。

输入：

```python
readColmapSceneInfo(path, images, eval, llffhold=8)
```

输出：

```python
SceneInfo(point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path)
```

核心流程：

```python
read cameras.bin / images.bin
if binary read fails:
    read cameras.txt / images.txt

cam_infos = readColmapCameras(...)
if eval:
    train = cameras where index % llffhold != 0
    test = cameras where index % llffhold == 0
else:
    train = all cameras
    test = []

if sparse/0/points3D.ply not exists:
    convert points3D.bin or points3D.txt to ply

pcd = fetchPly(points3D.ply)
return SceneInfo(...)
```

工程含义：这是 COLMAP 数据进入训练系统的第一道接口。后面的 `Scene` 不直接关心 COLMAP 文件细节，只消费 `SceneInfo`。

是否涉及 CUDA：不涉及，主要是文件 IO、相机解析和 PLY 解析。

### `scene/dataset_readers.py::readNerfSyntheticInfo(path, white_background, eval, extension=".png")`

作用：读取 Blender/NeRF synthetic 格式数据。

输入：

```python
readNerfSyntheticInfo(path, white_background, eval)
```

输出：同样返回 `SceneInfo`。

核心流程：

```python
train = readCamerasFromTransforms(path, "transforms_train.json", ...)
test = readCamerasFromTransforms(path, "transforms_test.json", ...)

if not eval:
    train += test
    test = []

if points3d.ply not exists:
    generate 100000 random points
    storePly(points3d.ply, xyz, rgb)

pcd = fetchPly(points3d.ply)
return SceneInfo(...)
```

工程含义：Blender 数据没有 COLMAP sparse point cloud，所以代码用随机点云启动优化。这和原始 3DGS 的 synthetic 处理方式一致。

### `utils/camera_utils.py::loadCam(args, id, cam_info, resolution_scale)`

作用：把 `CameraInfo` 转成训练/渲染阶段真正使用的 `Camera` 对象。

输入：

```python
loadCam(args, id, cam_info, resolution_scale)
```

输出：

```python
Camera(...)
```

核心流程：

```python
根据 args.resolution 决定图像缩放分辨率
resized_image_rgb = PILtoTorch(cam_info.image, resolution)
gt_image = resized_image_rgb[:3]
loaded_mask = alpha channel if exists
return Camera(...)
```

工程含义：这里决定训练图像是否降采样。大图默认会压到约 1600 像素宽，避免显存爆掉。

### `scene/cameras.py::Camera.__init__(...)`

作用：保存相机内外参、原始图像，并预计算 CUDA rasterizer 需要的矩阵。

输入：

```python
Camera(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, ...)
```

输出：一个带 GPU tensor 的 `Camera` 实例。

核心流程：

```python
self.original_image = image.to(data_device)
self.world_view_transform = getWorld2View2(R, T, trans, scale)
self.projection_matrix = getProjectionMatrix(...)
self.full_proj_transform = world_view_transform @ projection_matrix
self.camera_center = inverse(world_view_transform)[3, :3]
```

工程含义：renderer 不再做相机解析，它只读取 `world_view_transform`、`full_proj_transform`、`camera_center`、`FoVx/FoVy`。

是否涉及 CUDA：矩阵和图像会被放到 CUDA 上。

### `scene/__init__.py::Scene.__init__(args, gaussians, load_iteration=None, ...)`

作用：组织数据集、相机列表和 GaussianModel，是训练/渲染入口的场景上下文。

输入：

```python
Scene(args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], new_sh=0, load_vq=False)
```

输出：一个包含 `train_cameras`、`test_cameras`、`gaussians` 的 `Scene`。

核心流程：

```python
if source_path/sparse exists:
    scene_info = readColmapSceneInfo(...)
elif transforms_train.json exists:
    scene_info = readNerfSyntheticInfo(...)

cameraList_from_camInfos(...) 构造 train/test Camera

if load_vq:
    gaussians.load_vq(model_path)
elif new_sh != 0 and loaded_iter:
    gaussians.load_ply_sh(point_cloud.ply, new_sh)
elif loaded_iter:
    gaussians.load_ply(point_cloud.ply)
else:
    gaussians.create_from_pcd(scene_info.point_cloud, cameras_extent)
```

工程含义：`Scene` 是数据流分叉点。训练新模型、加载 PLY、加载降阶 SH、加载 VQ 压缩模型，都是在这里决定的。

## Gaussian 表示与数量管理

文件路径：`scene/gaussian_model.py`

核心类：`GaussianModel`

核心参数：

* `_xyz`: `[N, 3]`，Gaussian 中心。
* `_features_dc`: `[N, 1, 3]`，SH DC 项。
* `_features_rest`: `[N, K, 3]`，高阶 SH 系数。
* `_scaling`: `[N, 3]`，log-space scale，读取时 `exp`。
* `_rotation`: `[N, 4]`，四元数，读取时 normalize。
* `_opacity`: `[N, 1]`，logit-space opacity，读取时 sigmoid。
* `max_radii2D`: 每个 Gaussian 训练中见过的最大屏幕半径。
* `xyz_gradient_accum` / `denom`: screen-space 位置梯度统计，用于 densification。

### `GaussianModel.setup_functions()`

作用：集中定义参数激活函数和 covariance 构造函数。

输入：无显式输入，修改 `self`。

输出：给模型挂上多个函数属性。

核心流程：

```python
self.scaling_activation = torch.exp
self.scaling_inverse_activation = torch.log
self.opacity_activation = torch.sigmoid
self.inverse_opacity_activation = inverse_sigmoid
self.rotation_activation = normalize
self.covariance_activation = build_covariance_from_scaling_rotation
```

工程含义：模型内部保存的是 unconstrained 参数，读取属性时才映射到物理含义。例如 opacity 存 logit，scale 存 log。

### `GaussianModel.create_from_pcd(pcd, spatial_lr_scale)`

作用：从初始点云创建可训练 Gaussian 参数。

输入：

```python
create_from_pcd(pcd: BasicPointCloud, spatial_lr_scale: float)
```

输出：初始化 `self._xyz`、`self._features_dc`、`self._features_rest`、`self._scaling`、`self._rotation`、`self._opacity`。

核心流程：

```python
xyz = torch.tensor(pcd.points).cuda()
color = RGB2SH(pcd.colors)

features[:, :, 0] = color
features[:, :, 1:] = 0

dist2 = distCUDA2(points)
scales = log(sqrt(dist2)).repeat(1, 3)
rotation = identity quaternion
opacity = inverse_sigmoid(0.1)

wrap all tensors as nn.Parameter
```

工程含义：初始 Gaussian 数量等于点云点数。scale 用最近邻距离估计，避免初始 Gaussian 过大或过小。

是否涉及 CUDA：使用 `simple_knn._C.distCUDA2` 计算近邻距离。

### `GaussianModel.training_setup(training_args)`

作用：创建优化器，并为 densification 初始化统计缓冲。

输入：

```python
training_setup(training_args)
```

输出：

```python
self.optimizer
self.xyz_gradient_accum
self.denom
self.xyz_scheduler_args
```

核心流程：

```python
param_groups = [
    {"params": [_xyz], "lr": position_lr, "name": "xyz"},
    {"params": [_features_dc], "lr": feature_lr, "name": "f_dc"},
    {"params": [_features_rest], "lr": feature_lr / 20, "name": "f_rest"},
    {"params": [_opacity], "lr": opacity_lr, "name": "opacity"},
    {"params": [_scaling], "lr": scaling_lr, "name": "scaling"},
    {"params": [_rotation], "lr": rotation_lr, "name": "rotation"},
]
self.optimizer = AdamW(param_groups)
self.xyz_scheduler_args = get_expon_lr_func(...)
```

工程含义：每个属性独立学习率，且 param group 的 `name` 后面会被剪枝/拼接逻辑强依赖。新增属性时必须同步改 optimizer 管理函数。

### `GaussianModel.update_learning_rate(iteration)`

作用：只更新 `xyz` 参数组的指数衰减学习率。

输入：

```python
update_learning_rate(iteration)
```

输出：当前 `xyz` 学习率。

核心流程：

```python
for group in optimizer.param_groups:
    if group["name"] == "xyz":
        group["lr"] = xyz_scheduler_args(iteration)
```

工程含义：位置学习率随训练进行逐渐降低，其他属性学习率由 optimizer 参数组和外部 scheduler 控制。

### `GaussianModel.save_ply(path)`

作用：把当前 Gaussian 参数保存成 3DGS 标准 PLY。

输入：

```python
save_ply(path)
```

输出：`point_cloud.ply`。

核心流程：

```python
xyz = _xyz.detach().cpu().numpy()
f_dc = _features_dc.transpose(1, 2).flatten(...)
f_rest = _features_rest.transpose(1, 2).flatten(...)
opacity = _opacity
scale = _scaling
rotation = _rotation
attributes = concat(xyz, normals, f_dc, f_rest, opacity, scale, rotation)
PlyData.write(path)
```

工程含义：PLY 保存的是未激活的内部参数。例如 opacity 保存 logit，scale 保存 log scale，而不是 sigmoid/exp 后的值。

### `GaussianModel.load_ply(path)`

作用：从标准 3DGS PLY 恢复 Gaussian 参数。

输入：

```python
load_ply(path)
```

输出：恢复模型参数，并设置 `active_sh_degree = max_sh_degree`。

核心流程：

```python
read x/y/z
read f_dc_0..2
read all f_rest_*
assert f_rest count matches max_sh_degree
read opacity / scale_* / rot_*
wrap as nn.Parameter on cuda
```

工程含义：该函数要求 PLY 的 SH 维度和当前 `GaussianModel(sh_degree)` 一致。如果要加载低阶 SH，应使用 `load_ply_sh()`。

### `GaussianModel.load_ply_sh(path, new_sh)`

作用：加载高阶 PLY 时只保留低阶 SH 系数。

输入：

```python
load_ply_sh(path, new_sh)
```

输出：一个 SH degree 被截断后的 GaussianModel。

核心流程：

```python
read full f_rest
features_extra = reshape(N, 3, full_coeffs - 1)
features_extra = features_extra[:, :, : (new_sh + 1)^2 - 1]
self.active_sh_degree = new_sh
```

工程含义：这是 SH 蒸馏/降阶加载的重要基础，避免直接因为 SH 维度不匹配 assert。

### `GaussianModel.load_vq(path)`

作用：从 VecTree 量化目录反量化出完整 Gaussian 参数。

输入：

```python
load_vq(model_path)
```

期望目录：

```text
model_path/extreme_saving/
```

输出：恢复 `_xyz`、`_features_dc`、`_features_rest`、`_opacity`、`_scaling`、`_rotation`。

核心流程：

```python
dequantized_feats = load_vqgaussian(model_path/extreme_saving)
xyz = feats[:, 0:3]
features_dc = feats[:, 6:9]
features_rest = feats[:, 9:9+sh_dim]
opacity = feats[:, -8:-7]
scale = feats[:, -7:-4]
rotation = feats[:, -4:]
```

工程含义：当前实现是“加载时反量化成 dense tensor 再渲染”，不是直接在 codebook 压缩格式上渲染。

### `GaussianModel.cat_tensors_to_optimizer(tensors_dict)`

作用：densification 时把新 Gaussian 参数拼到已有参数后面，同时扩展 optimizer state。

输入：

```python
cat_tensors_to_optimizer({
    "xyz": new_xyz,
    "f_dc": new_features_dc,
    "f_rest": new_features_rest,
    "opacity": new_opacity,
    "scaling": new_scaling,
    "rotation": new_rotation,
})
```

输出：新的 `nn.Parameter` 字典。

核心流程：

```python
for each optimizer param group:
    old_param = group["params"][0]
    extension = tensors_dict[group["name"]]
    new_param = cat(old_param, extension, dim=0)
    optimizer state exp_avg / exp_avg_sq also cat zeros
```

工程含义：新增 Gaussian 时不仅要拼参数，还要保持 Adam state 维度一致，否则 optimizer step 会出错。

### `GaussianModel._prune_optimizer(mask)`

作用：剪枝时按 `mask` 保留参数，并同步裁剪 optimizer state。

输入：

```python
_prune_optimizer(mask)
```

这里的 `mask=True` 表示保留该 Gaussian。

输出：裁剪后的参数字典。

核心流程：

```python
for each optimizer param group:
    param = param[mask]
    state["exp_avg"] = state["exp_avg"][mask]
    state["exp_avg_sq"] = state["exp_avg_sq"][mask]
    replace group["params"][0] with new nn.Parameter
```

工程含义：这是 Gaussian 数量管理最关键的函数。剪枝不是简单删 `_xyz`，而是所有属性和 optimizer state 都必须同步删。

### `GaussianModel.prune_points(mask)`

作用：删除 mask 为 True 的 Gaussian。

输入：

```python
prune_points(mask)
```

输出：模型中剩余 Gaussian 数量减少。

核心流程：

```python
valid_points_mask = ~mask
optimizable_tensors = _prune_optimizer(valid_points_mask)
self._xyz = optimizable_tensors["xyz"]
...
self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
self.denom = self.denom[valid_points_mask]
self.max_radii2D = self.max_radii2D[valid_points_mask]
```

工程含义：外部传入的是“要删掉哪些点”，内部转换成“保留哪些点”。二次开发时这个语义容易写反。

### `GaussianModel.prune_gaussians(percent, import_score)`

作用：按重要性分数删除最低 `percent` 比例的 Gaussian。

输入：

```python
prune_gaussians(percent, import_score)
```

输出：低分 Gaussian 被删除。

核心流程：

```python
sorted_score = sort(import_score)
threshold = sorted_score[int(percent * (N - 1))]
prune_mask = import_score <= threshold
prune_points(prune_mask)
```

工程含义：`import_score` 越小越容易被删。LightGaussian 的 `v_important_score` 最终就是传给这个函数。

### `GaussianModel.add_densification_stats(viewspace_point_tensor, update_filter)`

作用：累积每个可见 Gaussian 的 screen-space 位置梯度。

输入：

```python
add_densification_stats(viewspace_point_tensor, visibility_filter)
```

输出：

```python
xyz_gradient_accum += norm(grad_xy)
denom += 1
```

核心流程：

```python
self.xyz_gradient_accum[visible] += norm(viewspace_point_tensor.grad[visible, :2])
self.denom[visible] += 1
```

工程含义：3DGS 用 2D 投影位置梯度判断哪些 Gaussian 对图像误差敏感。梯度大说明该区域表达不足，后续可能 clone/split。

### `GaussianModel.densify_and_clone(grads, grad_threshold, scene_extent)`

作用：对“小尺度但高梯度”的 Gaussian 直接复制。

输入：

```python
densify_and_clone(grads, grad_threshold, scene_extent)
```

输出：新增一批 Gaussian。

核心流程：

```python
selected = norm(grads) >= grad_threshold
selected &= max_scaling <= percent_dense * scene_extent
new_params = old_params[selected]
densification_postfix(new_params)
```

工程含义：小 Gaussian 说明已经是局部细节，直接 clone 可以增加表达密度。

### `GaussianModel.densify_and_split(grads, grad_threshold, scene_extent, N=2)`

作用：对“大尺度且高梯度”的 Gaussian 拆分成多个更小 Gaussian。

输入：

```python
densify_and_split(grads, grad_threshold, scene_extent, N=2)
```

输出：新增拆分点，并删除原来的大 Gaussian。

核心流程：

```python
selected = grads >= threshold
selected &= max_scaling > percent_dense * scene_extent

samples = normal(mean=0, std=scaling)
new_xyz = rotation @ samples + old_xyz
new_scaling = log(old_scaling / (0.8 * N))
densification_postfix(new_gaussians)
prune_points(original_selected_mask)
```

工程含义：大 Gaussian 覆盖范围广，误差高时更适合分裂，而不是简单复制。

### `GaussianModel.densify_and_prune(max_grad, min_opacity, extent, max_screen_size)`

作用：原始 3DGS 训练阶段的密化与基础剪枝。

输入：

```python
densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
```

输出：Gaussian 数量可能先增加再减少。

核心流程：

```python
grads = xyz_gradient_accum / denom
densify_and_clone(grads, ...)
densify_and_split(grads, ...)

prune_mask = opacity < min_opacity
if max_screen_size:
    prune too large screen-space / world-space gaussians
prune_points(prune_mask)
```

工程含义：这不是 LightGaussian 的全局重要性剪枝，而是原始 3DGS 训练中为了稳定几何和透明度做的维护性剪枝。

## 渲染流程与 CUDA Splatting

### `gaussian_renderer/__init__.py::render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None)`

作用：给定相机和 GaussianModel，渲染一张图，并保留训练需要的可见性和梯度信息。

输入：

```python
render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None)
```

输出：

```python
{
    "render": rendered_image,
    "viewspace_points": screenspace_points,
    "visibility_filter": radii > 0,
    "radii": radii,
}
```

核心流程：

```python
screenspace_points = zeros_like(pc.get_xyz, requires_grad=True)

raster_settings = GaussianRasterizationSettings(
    image_height, image_width, tanfovx, tanfovy,
    viewmatrix, projmatrix, sh_degree, campos,
    f_count=False
)

if pipe.compute_cov3D_python:
    cov3D_precomp = pc.get_covariance(...)
else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

if pipe.convert_SHs_python:
    colors_precomp = eval_sh(...)
else:
    shs = pc.get_features

rendered_image, radii = rasterizer(...)
```

工程含义：`screenspace_points` 本身不是直接参与投影计算的真实 2D 坐标，但它作为可求导占位张量接收 CUDA backward 回传的 screen-space 梯度，用于 densification。

是否涉及 CUDA：核心渲染和反传在 `diff_gaussian_rasterization` CUDA 扩展内完成。

### `gaussian_renderer/__init__.py::count_render(...)`

作用：在普通渲染之外，额外统计每个 Gaussian 的使用次数和重要性分数。

输入：和 `render()` 基本一致。

输出：

```python
{
    "render": rendered_image,
    "viewspace_points": screenspace_points,
    "visibility_filter": radii > 0,
    "radii": radii,
    "gaussians_count": gaussians_count,
    "important_score": important_score,
}
```

核心流程：

```python
raster_settings = GaussianRasterizationSettings(..., f_count=True)
rasterizer = GaussianRasterizer(raster_settings)
gaussians_count, important_score, image, radii = rasterizer(...)
```

工程含义：这是 LightGaussian 计算全局重要性的入口。训练正常反传用 `render()`，剪枝统计用 `count_render()`。

### `diff_gaussian_rasterization::_RasterizeGaussians.forward(...)`

文件路径：`submodules/compress-diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`

作用：PyTorch autograd wrapper 的 forward，把 Python tensor 传给 C++/CUDA。

输入：Gaussian 参数、相机矩阵、SH 或预计算颜色、raster settings。

输出：

* 普通模式：`color, radii`
* 计数模式：`gaussians_count, important_score, color, radii`

核心流程：

```python
if raster_settings.f_count:
    return _C.count_gaussians(...)
else:
    return _C.rasterize_gaussians(...)
```

工程含义：LightGaussian 没有在 Python 里遍历像素统计贡献，而是在 CUDA rasterizer 的像素混合循环里顺手统计，避免 Python 侧巨大开销。

### `diff_gaussian_rasterization::_RasterizeGaussians.backward(...)`

作用：调用 CUDA backward，返回对 Gaussian 参数的梯度。

输入：

```python
grad_out_color
```

输出：

```python
grad_means3D, grad_means2D, grad_sh, grad_opacities, grad_scales, grad_rotations, ...
```

核心流程：

```python
restore saved tensors
call _C.rasterize_gaussians_backward(...)
return grads in PyTorch expected order
```

工程含义：训练中 Gaussian 的位置、SH、opacity、scale、rotation 都能通过 rasterizer 反传优化。

### `rasterize_points.cu::RasterizeGaussiansCUDA(...)`

作用：普通 CUDA forward 的 C++ host 封装。

输入：background、means3D、colors/SH、opacity、scale、rotation、view/proj matrix、FoV、image size。

输出：

```cpp
rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer
```

核心流程：

```cpp
allocate out_color and radii
allocate geometry / binning / image buffers
call CudaRasterizer::Rasterizer::forward(...)
return buffers for backward
```

工程含义：`geomBuffer`、`binningBuffer`、`imgBuffer` 会在 backward 复用，避免重新计算中间结构。

### `rasterize_points.cu::CountGaussiansCUDA(...)`

作用：LightGaussian 修改版 forward，返回每个 Gaussian 的统计信息。

输出：

```cpp
gaussians_count, important_score, rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer
```

核心流程：

```cpp
gaussians_count = zeros(P)
important_score = zeros(P)
call CudaRasterizer::Rasterizer::forwardCount(...)
```

工程含义：它和普通 forward 共享大部分 rasterizer 流程，只在最终 pixel blending kernel 换成带统计的版本。

### `cuda_rasterizer/forward.cu::preprocessCUDA(...)`

作用：每个 Gaussian 并行完成投影前处理。

输入：Gaussian 中心、scale、rotation、opacity、SH、相机矩阵。

输出：

* `radii`: 每个 Gaussian 的屏幕半径。
* `points_xy_image`: 投影到图像上的 2D 坐标。
* `depths`: 深度。
* `cov3Ds`: 3D covariance。
* `rgb`: SH 转出的颜色。
* `conic_opacity`: 2D Gaussian 逆协方差和 opacity。
* `tiles_touched`: 覆盖多少 tile。

核心流程：

```cpp
if not in_frustum:
    return

project mean to screen
compute cov3D from scale/rotation if needed
compute cov2D with projection Jacobian
invert cov2D to conic
estimate radius from eigenvalue
find touched tile rectangle
compute RGB from SH if needed
write helper buffers
```

工程含义：后续 tile sorting 和 splatting 都依赖这里产生的中间结果。

### `cuda_rasterizer/rasterizer_impl.cu::duplicateWithKeys(...)`

作用：把每个 Gaussian 复制到它覆盖的所有 tile 中，并生成排序 key。

输入：`points_xy`、`depths`、`offsets`、`radii`、tile grid。

输出：

* `gaussian_keys_unsorted`
* `gaussian_values_unsorted`

核心流程：

```cpp
for each gaussian:
    if radius > 0:
        rect = covered tiles
        for each tile in rect:
            key = tile_id << 32 | depth_bits
            value = gaussian_id
```

工程含义：后续 CUB radix sort 会按 tile 和 depth 排序，使每个 tile 内 Gaussian 可以 front-to-back 混合。

### `cuda_rasterizer/forward.cu::renderCUDA(...)`

作用：普通 splatting kernel，每个 tile 一个 block，每个线程处理一个像素。

输入：tile ranges、排序后的 Gaussian list、2D mean、颜色、conic opacity、背景色。

输出：`out_color`。

核心流程：

```cpp
for each tile block:
    load gaussian ids in batches into shared memory
    for each pixel thread:
        for each gaussian in sorted tile range:
            power = -0.5 * d^T conic d
            alpha = opacity * exp(power)
            C += color * alpha * T
            T *= 1 - alpha
            if T < 0.0001:
                stop
```

工程含义：这是 3DGS 实时渲染的核心。按 tile 分块和 shared memory 批加载是性能关键。

### `cuda_rasterizer/forward.cu::renderCUDA_count(...)`

作用：LightGaussian 增强版 splatting kernel，在混合时顺带统计 Gaussian 贡献。

输入：和 `renderCUDA()` 基本一致，额外包含 `gaussian_count`、`important_score`。

输出：图像、每个 Gaussian 的 count 和 importance。

核心差异：

```cpp
if alpha is valid and pixel not done:
    gaussian_count[collected_id[j]]++;
    important_score[collected_id[j]] += con_o.w;
```

工程含义：`gaussian_count` 是参与次数，`important_score` 当前实现累加的是 opacity 项 `con_o.w`。它不是完整 alpha/T 贡献，因此是一个近似重要性指标。

注意：这里没有 `atomicAdd`，多线程同时更新同一个 Gaussian 时存在数据竞争，统计值可能非完全确定。

## 训练入口函数

### `train_densify_prune.py::training(...)`

作用：从数据集初始化 Gaussian，执行原始 3DGS 训练，并在指定 iteration 插入 LightGaussian 剪枝。

输入：

```python
training(dataset, opt, pipe, testing_iterations, saving_iterations,
         checkpoint_iterations, checkpoint, debug_from, args)
```

输出：

* `point_cloud/iteration_*/point_cloud.ply`
* `chkpnt*.pth`
* `metric.csv`
* `imp_score.npz`

核心流程：

```python
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)

for iteration in training range:
    network_gui.try_connect()
    gaussians.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    viewpoint_cam = random training camera
    render_pkg = render(...)
    loss = L1 + DSSIM
    loss.backward()

    if iteration < densify_until_iter:
        update max_radii2D
        add_densification_stats()
        densify_and_prune() periodically
        reset_opacity() periodically

    if iteration in args.prune_iterations:
        gaussian_list, imp_list = prune_list(...)
        v_list = calculate_v_imp_score(...)
        gaussians.prune_gaussians(...)

    optimizer.step()
    save / checkpoint / evaluate if needed
```

工程含义：这是“训练中剪枝”的路线。它把原始 3DGS 的 densify/prune 与 LightGaussian 的全局重要性剪枝放在同一个训练循环里。

是否涉及 CUDA：每轮 render/backward、剪枝统计里的 count_render 都走 CUDA rasterizer。

### `prune_finetune.py::training(...)`

作用：加载已有 checkpoint 或 PLY，先剪枝再 fine-tune 恢复质量。

输入：

```python
training(dataset, opt, pipe, testing_iterations, saving_iterations,
         checkpoint_iterations, checkpoint, debug_from, args)
```

关键参数：

* `--start_checkpoint`: 从 `.pth` 恢复完整训练状态。
* `--start_pointcloud`: 从 `point_cloud.ply` 加载 Gaussian。
* `--prune_iterations`: 哪些 iteration 执行剪枝。
* `--prune_percent`: 每次剪掉多少比例。
* `--prune_type`: 用哪种分数剪枝。

输出：剪枝恢复后的 PLY/checkpoint/metrics/importance。

核心流程：

```python
if checkpoint:
    gaussians.training_setup(opt)
    gaussians.restore(torch.load(checkpoint), opt)
elif start_pointcloud:
    gaussians.load_ply(start_pointcloud)
    gaussians.training_setup(opt)
else:
    raise

for iteration:
    render()
    loss = L1 + DSSIM against GT image
    loss.backward()

    if iteration in prune_iterations:
        gaussian_list, imp_list = prune_list(...)
        if prune_type == "important_score":
            score = imp_list
        elif prune_type == "v_important_score":
            score = calculate_v_imp_score(...)
        elif prune_type == "count":
            score = gaussian_list
        elif prune_type == "opacity":
            score = gaussians.get_opacity
        gaussians.prune_gaussians(percent, score)

    optimizer.step()
```

工程含义：这是论文默认更常用的“先训练原始模型，再剪枝恢复”的路线。它比训练中剪枝更容易控制压缩比例和质量恢复。

### `distill_train.py::training(...)`

作用：把高阶 SH teacher 模型蒸馏到低阶 SH student 模型。

输入：

```python
training(args, dataset, opt, pipe, testing_iterations, saving_iterations,
         checkpoint_iterations, checkpoint, debug_from, new_max_sh)
```

关键参数：

* `--teacher_model`: teacher checkpoint。
* `--start_checkpoint`: student 初始 checkpoint。
* `--new_max_sh`: student 的目标 SH degree。
* `--augmented_view`: 是否使用扰动虚拟视角。
* `--enable_covariance`: 是否允许 student 更新 scale/rotation。
* `--enable_opacity`: 是否允许 student 更新 opacity。

输出：低阶 SH student 的 PLY/checkpoint/importance。

核心流程：

```python
teacher_gaussians = GaussianModel(old_sh_degree)
student_gaussians = GaussianModel(old_sh_degree)

teacher_gaussians.restore(teacher_model)
student_gaussians.restore(start_checkpoint)
student_gaussians.max_sh_degree = new_max_sh
student_gaussians.onedownSHdegree()

if not enable_covariance:
    freeze scaling and rotation
if not enable_opacity:
    freeze opacity

for iteration:
    choose train camera
    if augmented_view:
        perturb camera pose
    student_image = render(student)
    teacher_image = render(teacher).detach()
    loss = L1 + DSSIM between student_image and teacher_image
    loss.backward()
    optimizer.step()
```

工程含义：这里的监督不是 GT 图像，而是 teacher 渲染结果。这样可以让低阶 SH student 尽量保持 teacher 的视角相关颜色表达。

### `utils/logger_utils.py::training_report(...)`

作用：训练中定期评估并记录指标。

输入：

```python
training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                testing_iterations, scene, renderFunc, renderArgs)
```

输出：

* TensorBoard scalar/image。
* `metric.csv`。
* 控制台 PSNR/SSIM/LPIPS。

核心流程：

```python
if iteration in testing_iterations:
    for each test camera:
        image = renderFunc(camera, scene.gaussians, ...)
        compare with GT
        accumulate L1 / PSNR / SSIM / LPIPS
    append row to metric.csv
```

工程含义：压缩方法最终要看质量和文件大小，这个函数把指标、耗时、PLY 文件大小一起落到 CSV，方便做实验表。

## 剪枝函数

文件路径：`prune.py`

`prune.py` 的两个核心函数是 LightGaussian 剪枝策略的入口：一个负责“统计每个 Gaussian 在所有训练视角中的贡献”，另一个负责“把贡献分数和 Gaussian 体积结合，得到最终剪枝分数”。

### `prune_list(gaussians, scene, pipe, background)`

作用：遍历所有训练相机，用修改版 CUDA rasterizer 统计每个 Gaussian 的全局可见/贡献情况。

输入：

```python
prune_list(gaussians, scene, pipe, background)
```

输出：

```python
gaussian_list, imp_list
```

含义：

* `gaussian_list`: 每个 Gaussian 被参与像素 splatting 的次数累计。
* `imp_list`: 每个 Gaussian 的重要性累计分数，来自 `count_render()` 返回的 `important_score`。

核心流程：

```python
viewpoint_stack = scene.getTrainCameras().copy()

for each training camera:
    pkg = count_render(cam, gaussians, pipe, background)
    gaussian_list += pkg["gaussians_count"]
    imp_list += pkg["important_score"]
```

工程含义：这里的关键不是普通 `render()`，而是 `count_render()`。它会走修改过的 CUDA rasterizer，在渲染每个视角时额外统计每个 Gaussian 对像素混合的参与情况。

本质上它在回答：

```text
每个 Gaussian 在所有训练视角中到底有多常被用到？贡献大不大？
```

### `calculate_v_imp_score(gaussians, imp_list, v_pow)`

作用：在 `imp_list` 的基础上加入 Gaussian 体积因子，得到 LightGaussian 的 volume-aware importance score。

输入：

```python
calculate_v_imp_score(gaussians, imp_list, v_pow)
```

输出：

```python
v_list
```

核心逻辑：

```python
volume = torch.prod(gaussians.get_scaling, dim=1)
sorted_volume, _ = torch.sort(volume, descending=True)
kth_percent_largest = sorted_volume[int(len(volume) * 0.9)]

v_list = torch.pow(volume / kth_percent_largest, v_pow)
v_list = v_list * imp_list
```

工程含义：

* `imp_list` 表示渲染贡献。
* `volume = sx * sy * sz` 表示 Gaussian 的空间体积。
* `kth_percent_largest` 用第 90% 位置的体积做归一化基准。
* `v_pow` 控制体积对最终分数的影响强度。
* 最终 `v_list` 越小，越容易被剪掉。

为什么要加 volume：普通 `imp_list` 只看渲染贡献，容易误删一些覆盖范围大但单次贡献不突出的 Gaussian。LightGaussian 通过体积加权，让空间覆盖更大的 Gaussian 获得一定保护。

最终调用：

```python
gaussian_list, imp_list = prune_list(...)
v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
gaussians.prune_gaussians(prune_percent, v_list)
```

## VecTree / VQ 压缩

### `vectree/vectree.py::Quantization.__init__(opt)`

作用：读取 PLY 特征，构造 VQ codebook，并准备重要性路径和保存路径。

输入：

```python
Quantization(opt)
```

关键参数：

* `--input_path`: 待量化的 `point_cloud.ply`。
* `--important_score_npz_path`: 包含 `imp_score.npz` 的目录。
* `--vq_ratio`: 进入 VQ 的 Gaussian 比例。
* `--codebook_size`: codebook 大小，默认 8192。
* `--sh_degree`: SH degree，决定 SH 特征维度。

输出：初始化后的量化器对象。

核心流程：

```python
self.feats_bak = read_ply_data(input_path)
self.feats = self.feats_bak[:, 6:6+self.sh_dim]
self.model_vq = VectorQuantize(dim=self.feats.shape[1], codebook_size=...)
```

工程含义：VecTree 主要量化 SH 相关特征，`xyz` 和最后的 `opacity + scale + rotation` 单独保存。

### `Quantization.quantize()`

作用：根据重要性选择哪些 Gaussian 保留原始 SH，哪些进入 VQ。

输入：使用对象初始化时保存的 PLY 特征和重要性分数。

输出：`extreme_saving/` 压缩文件集合。

核心流程：

```python
if no_IS:
    importance = ones(N)
else:
    importance = load imp_score.npz

large_val, large_index = topk(importance, k=N * (1 - vq_ratio))
non_vq_mask[large_index] = True
vq_mask = ~non_vq_mask

for i in iteration_num:
    sample VQ_CHUNK features from vq_mask
    weight = importance of sampled features
    model_vq(feature, weight=weight)
    replace dead/low-use code with important samples

fully_vq_reformat()
```

工程含义：重要 Gaussian 的 SH 不量化，低重要 Gaussian 的 SH 用 codebook 表示。这是质量和压缩率之间的折中。

### `Quantization.calc_vector_quantized_feature()`

作用：把所有需要量化的 SH 特征映射到 codebook index，并得到量化后的特征。

输入：`self.feats`。

输出：

```python
all_feat, all_indice
```

核心流程：

```python
for chunk in feats:
    feat, indices, commit = model_vq(chunk)
    append feat and indices
```

工程含义：分 chunk 是为了避免一次性对所有 Gaussian 做 VQ 导致显存过高。

### `Quantization.fully_vq_reformat()`

作用：把 VQ 结果保存成 LightGaussian 的压缩目录结构。

输出目录：

```text
extreme_saving/
  metadata.npz
  vq_indexs.npz
  codebook.npz
  non_vq_mask.npz
  non_vq_feats.npz
  other_attribute.npz
  xyz.npz
```

核心流程：

```python
save metadata
pack vq indices into bits
save codebook in float16
pack non_vq_mask
save non_vq_feats in half
save other_attribute in half
save xyz
zip extreme_saving
```

工程含义：真正的磁盘压缩发生在这里。`vq_indexs` 被 bit-pack，codebook 和非量化特征使用 half 存储。

### `vectree/utils.py::load_vqgaussian(path, device="cuda")`

作用：从 `extreme_saving/` 反量化回完整 Gaussian 属性矩阵。

输入：

```python
load_vqgaussian(path)
```

输出：

```python
full_feats: [N, input_pc_dim]
```

核心流程：

```python
load metadata
unpack non_vq_mask
load codebook
unpack vq_indexs and convert binary to decimal
load non_vq_feats
load xyz and other_attribute

full_feats[:, 0:3] = xyz
full_feats[:, -8:] = other_attribute
full_feats[vq_mask, 6:6+codebook_dim] = codebook[vq_indexs]
full_feats[non_vq_mask, 6:6+codebook_dim] = non_vq_feats
```

工程含义：它负责把压缩格式还原成 `GaussianModel.load_vq()` 能理解的 dense 属性矩阵。

### `vectree/vq.py::VectorQuantize.forward(x, weight=None, verbose=False)`

作用：执行向量量化，把连续特征替换成 codebook 中最近的 embedding。

输入：

```python
VectorQuantize.forward(x, weight=None)
```

输出：

```python
quantize, embed_ind, loss
```

核心流程：

```python
x = project_in(x)
quantize, embed_ind = codebook(x, weight)
if training:
    quantize = x + (quantize - x).detach()
    loss = commitment loss
quantize = project_out(quantize)
```

工程含义：这是标准 VQ 的 straight-through estimator。LightGaussian 在 `weight` 中传入重要性，使 codebook 更新更偏向重要样本。

### `vectree/vq.py::EuclideanCodebook.forward(x, weight=None, verbose=False)`

作用：维护欧式距离 codebook，并用 EMA 更新 embedding。

输入：待量化特征 `x`，可选重要性权重 `weight`。

输出：

```python
quantize, embed_ind
```

核心流程：

```python
dist = -torch.cdist(flatten, embed)
embed_ind = argmax(dist)
quantize = embedding(embed_ind)

if training:
    cluster_size = onehot count weighted by importance
    embed_sum = weighted feature sum
    EMA update cluster_size and embed
```

工程含义：`weight` 会影响 codebook 的聚类中心更新，重要 Gaussian 对 codebook 有更大影响。

## 渲染与评估入口

### `render.py::render_sets(...)`

作用：加载模型并渲染 train/test 图像集合。

输入：

```python
render_sets(dataset, iteration, pipeline, skip_train, skip_test, load_vq)
```

输出：

```text
model_path/train/ours_ITER/renders/*.png
model_path/test/ours_ITER/renders/*.png
```

核心流程：

```python
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq)
if not skip_train:
    render_set(train cameras)
if not skip_test:
    render_set(test cameras)
```

工程含义：普通 PLY 和 VecTree 压缩模型都走同一套 `Scene` 加载逻辑。

### `render_video.py::render_video(...)`

作用：沿生成的相机轨迹渲染视频帧。

输入：

```python
render_video(model_path, iteration, views, gaussians, pipeline, background)
```

输出：

```text
model_path/video/ours_ITER/*.png
```

核心流程：

```python
for pose in generate_ellipse_path(views, n_frames=600):
    update view.world_view_transform
    update full_proj_transform and camera_center
    rendering = render(view, gaussians, pipeline, background)["render"]
    save png
```

工程含义：这里没有重新构造 Camera，而是复用一个 view 对象并修改其矩阵。改轨迹时主要看 `utils/pose_utils.py`。

# 4. 关键算法机制拆解

## Gaussian 参数化

代码位置：`scene/gaussian_model.py::setup_functions()`

核心公式：

```text
scale = exp(_scaling)
opacity = sigmoid(_opacity)
rotation = normalize(_rotation)
L = build_scaling_rotation(scale, rotation)
covariance = L L^T
```

工程解释：优化器更新的是无约束参数，渲染时通过 activation 转成合法的尺度、透明度和旋转。

## 3D Gaussian 投影到 2D

代码位置：`cuda_rasterizer/forward.cu::computeCov2D()`

核心逻辑：

```text
cov2D = J^T W^T cov3D W J
conic = inverse(cov2D)
radius = ceil(3 * sqrt(max_eigenvalue(cov2D)))
```

工程解释：3D covariance 经过相机外参和投影雅可比变换成屏幕空间 2D Gaussian。`conic` 是逆协方差，像素混合时用它快速计算 Gaussian falloff。

## Alpha blending

代码位置：`cuda_rasterizer/forward.cu::renderCUDA()`

核心逻辑：

```text
power = -0.5 * d^T conic d
alpha = min(0.99, opacity * exp(power))
C += color * alpha * T
T *= (1 - alpha)
```

工程解释：每个像素按 depth 排序后的 Gaussian 前向合成。`T` 是当前透射率，越靠前且 alpha 越大，对颜色贡献越大。

## Densification

代码位置：`GaussianModel.add_densification_stats()`、`densify_and_clone()`、`densify_and_split()`

核心逻辑：

```text
screen-space grad high + small Gaussian -> clone
screen-space grad high + large Gaussian -> split
opacity too low or screen/world size too large -> prune
```

工程解释：这是原始 3DGS 的自适应点数管理。LightGaussian 的剪枝发生在这个基础之上。

## Global significance

代码位置：`count_render()`、`renderCUDA_count()`、`prune_list()`

核心逻辑：

```text
for each training camera:
    render with f_count=True
    accumulate per-Gaussian count and importance
```

工程解释：LightGaussian 通过所有训练视角累计贡献，避免只根据单视角或 opacity 做局部判断。

## Volume-aware importance

代码位置：`calculate_v_imp_score()`

核心逻辑：

```text
v_score = imp_score * (volume / volume_ref) ^ v_pow
```

工程解释：引入体积项后，大范围覆盖的 Gaussian 不会因为单像素贡献较弱而被过早删除。

## SH distillation

代码位置：`distill_train.py::training()`

核心逻辑：

```text
teacher_image = render(high_SH_model).detach()
student_image = render(low_SH_model)
loss = L1(student_image, teacher_image) + DSSIM
```

工程解释：teacher 提供高阶 SH 的视角相关颜色目标，student 用更少 SH 系数拟合它。

## VecTree / VQ

代码位置：`vectree/vectree.py::quantize()`、`vectree/vq.py`

核心逻辑：

```text
important Gaussian -> keep raw SH
less important Gaussian -> replace SH by codebook index
```

工程解释：这一步主要压缩存储体积。当前渲染前仍会反量化为 dense Gaussian。

# 5. 项目运行流程（非常重要）

## 1. 数据如何准备

COLMAP 数据：

```bash
python convert.py -s PATH/TO/DATASET --resize
```

背后调用：

* `convert.py` 调 COLMAP feature extraction、matching、mapper、image undistorter。
* `readColmapSceneInfo()` 读取 `sparse/0`。
* `readColmapCameras()` 解析内外参。
* `fetchPly()` 读取初始点云。
* `cameraList_from_camInfos()` 构造训练和测试 Camera。

Blender 数据：

* 需要 `transforms_train.json` 和 `transforms_test.json`。
* `readNerfSyntheticInfo()` 读取相机。
* 如果没有 `points3d.ply`，随机生成 100k 初始点。

## 2. 如何启动训练

从头训练并剪枝：

```bash
bash scripts/run_train_densify_prune.sh
```

实际入口：

```bash
python train_densify_prune.py \
  -s DATASET \
  -m OUTPUT \
  --prune_percent 0.6 \
  --prune_iterations 20000 \
  --v_pow 0.1 \
  --eval
```

背后模块调用：

```text
train_densify_prune.training()
-> Scene(...)
-> GaussianModel.create_from_pcd()
-> GaussianModel.training_setup()
-> render()
-> GaussianModel.add_densification_stats()
-> GaussianModel.densify_and_prune()
-> prune_list()
-> calculate_v_imp_score()
-> GaussianModel.prune_gaussians()
-> Scene.save()
```

## 3. 如何进行 checkpoint 剪枝恢复

```bash
bash scripts/run_prune_finetune.sh
```

实际入口：

```bash
python prune_finetune.py \
  -s DATASET \
  -m OUTPUT \
  --start_checkpoint PATH/TO/chkpnt30000.pth \
  --prune_percent 0.66 \
  --prune_type v_important_score \
  --iteration 35000
```

背后模块调用：

```text
prune_finetune.training()
-> GaussianModel.restore() or load_ply()
-> render()
-> prune_list()
-> calculate_v_imp_score()
-> prune_gaussians()
-> continue optimizer.step()
```

## 4. 如何进行 SH 蒸馏

```bash
bash scripts/run_distill_finetune.sh
```

实际入口：

```bash
python distill_train.py \
  -s DATASET \
  -m OUTPUT \
  --teacher_model TEACHER/chkpnt30000.pth \
  --start_checkpoint STUDENT/chkpnt30000.pth \
  --new_max_sh 2 \
  --augmented_view \
  --enable_covariance
```

背后模块调用：

```text
distill_train.training()
-> teacher_gaussians.restore()
-> student_gaussians.restore()
-> student_gaussians.onedownSHdegree()
-> optional gaussian_poses()
-> render(student)
-> render(teacher)
-> optimize student with teacher image target
```

## 5. 如何进行 VecTree 量化

```bash
bash scripts/run_vectree_quantize.sh
```

实际入口：

```bash
python vectree/vectree.py \
  --important_score_npz_path PATH/TO/imp_score_dir \
  --input_path PATH/TO/point_cloud.ply \
  --save_path OUTPUT \
  --vq_ratio 0.6 \
  --codebook_size 8192
```

背后模块调用：

```text
Quantization.__init__()
-> read_ply_data()
-> Quantization.quantize()
-> VectorQuantize.forward()
-> EuclideanCodebook.forward()
-> fully_vq_reformat()
-> load_vqgaussian()
-> write_ply_data()
```

## 6. 如何渲染

普通 PLY/checkpoint：

```bash
python render.py -s DATASET -m MODEL --iteration -1 --skip_train
```

视频轨迹：

```bash
python render_video.py -s DATASET -m MODEL --skip_train --skip_test --video
```

VecTree 量化结果：

```bash
python render_video.py -s DATASET -m MODEL --load_vq --video
```

背后模块调用：

```text
render_sets()
-> Scene(load_iteration=-1, load_vq=...)
-> GaussianModel.load_ply() or load_vq()
-> render_set() or render_video()
-> render()
-> CUDA rasterizer
```

# 6. 性能优化点

## CUDA rasterizer 加速

相关函数：

* `preprocessCUDA()`
* `duplicateWithKeys()`
* `identifyTileRanges()`
* `renderCUDA()`
* `renderCUDA_count()`
* `BACKWARD::render()`
* `BACKWARD::preprocess()`

为什么快：

* 每个 Gaussian 并行做投影、covariance、半径和 tile 范围估计。
* 每个 Gaussian 被展开到覆盖的 tile，使用 CUB prefix sum 和 radix sort 组织 tile workload。
* 每个 tile 一个 CUDA block，每个线程处理一个像素。
* shared memory 分批加载 Gaussian 数据，减少全局内存访问。
* front-to-back alpha blending 支持透射率早停。

## Gaussian 数量减少带来的收益

相关函数：

* `prune_list()`
* `calculate_v_imp_score()`
* `GaussianModel.prune_gaussians()`
* `GaussianModel.prune_points()`

为什么快：

* Gaussian 数量 `N` 减少后，`preprocessCUDA()` 的线程数量减少。
* tile duplication 的实例数减少。
* radix sort 的输入长度减少。
* 每个 tile/pixel 需要混合的 Gaussian 数减少。

## SH 降阶带来的收益

相关函数：

* `distill_train.training()`
* `GaussianModel.load_ply_sh()`
* `GaussianModel.onedownSHdegree()`

为什么快/省：

* SH degree 从 3 到 2，颜色系数数量减少。
* PLY 文件里的 `f_rest_*` 属性减少。
* SH 计算和内存带宽压力下降。

## VecTree 量化带来的收益

相关函数：

* `Quantization.quantize()`
* `fully_vq_reformat()`
* `load_vqgaussian()`

为什么省：

* 大量 SH 特征由 codebook index 表示。
* index 被 bit-pack。
* codebook、非量化 SH、其他属性部分使用 half 保存。

注意：当前代码渲染前会 `load_vq()` 反量化成 dense tensor，所以 VecTree 主要减少存储和传输体积，不是直接减少 rasterizer 运行时计算。

# 7. 可扩展性分析（加分项）

## 新剪枝策略

优先改：

* `prune.py::calculate_v_imp_score()`
* `prune_finetune.py` 中 `if args.prune_type == ...` 分支
* `GaussianModel.prune_gaussians()`

适合做的研究：

* 把 `important_score += opacity` 改成 alpha/T 加权。
* 对 `gaussian_list` 做视角归一化。
* 结合 LPIPS 梯度或图像残差构造重要性。

## 新全局统计指标

优先改：

* `gaussian_renderer.count_render()`
* `_RasterizeGaussians.forward_count()`
* `CountGaussiansCUDA()`
* `forwardCount()`
* `renderCUDA_count()`

注意：Python wrapper、C++ binding、CUDA kernel 参数顺序必须同时改。

## 新 Gaussian 属性

优先改：

* `GaussianModel.__init__()`
* `training_setup()`
* `save_ply()` / `load_ply()`
* `cat_tensors_to_optimizer()`
* `_prune_optimizer()`
* renderer Python wrapper 和 CUDA 参数列表

注意：所有属性第一维必须和 Gaussian 数量 `N` 对齐，否则 densify/prune 会失配。

## 新量化策略

优先改：

* `vectree/vectree.py::quantize()`
* `fully_vq_reformat()`
* `vectree/utils.py::load_vqgaussian()`

适合尝试：

* 分属性 codebook。
* 分层 codebook。
* 对 opacity/scale/rotation 也做量化。
* 熵编码替代当前 zip。

## 新渲染轨迹

优先改：

* `utils/pose_utils.py`
* `render_video.py::render_video()`
* `render_video.py::render_circular_video()`

工程建议：新增轨迹函数时保持输出 pose 格式和现有 `generate_ellipse_path()` 一致，这样只需要替换 enumerate 的轨迹来源。

# 8. 常见坑 / 难点分析

* `submodules/compress-diff-gaussian-rasterization` 是修改版，不是原始 upstream，必须用本仓库 submodule 编译安装。
* `renderCUDA_count()` 里 `gaussian_count++` 和 `important_score += ...` 没有 `atomicAdd`，多线程写同一 Gaussian 时存在数据竞争，统计值可能是近似且非完全确定。
* `prune_list()` 会遍历所有训练相机做 `count_render()`，大场景上很耗时，建议只在少数 prune iteration 调用。
* `GaussianModel.prune_points()` 默认 optimizer 已初始化，不要在 `training_setup()` 前直接调用。
* `prune_points(mask)` 的 mask 语义是 True 表示删除；`_prune_optimizer(mask)` 的 mask 语义是 True 表示保留，二次开发时容易写反。
* `load_ply()` 会 assert SH 维度匹配。degree 不一致时要用 `load_ply_sh()`，或确保 `--sh_degree` 对齐。
* `Scene(load_vq=True)` 期望目录是 `MODEL/extreme_saving/`，不是直接读 `extreme_saving.zip`。
* `scripts/*.sh` 里大量 `PATH/TO/...` 和日志目录需要手动改，README 命令不能直接复制运行。
* 训练脚本参数真实名称是 `--iterations`，脚本里有些地方写 `--iteration` 是 argparse abbreviation，建议二次开发时改成完整参数名。
* `full_eval.py` 还引用了原始 3DGS 的 `train.py`，当前仓库没有这个文件，评估脚本需要按 LightGaussian 入口改造。
* VecTree 当前压缩 SH 特征为主，渲染前会反量化为 dense Gaussian，不要误以为它已经实现了 codebook 直接渲染。
* `GaussianModel` 的 optimizer param group 名称和 `_prune_optimizer()`、`cat_tensors_to_optimizer()` 强耦合。新增属性时必须同时维护这几处。
* PLY 属性顺序和 `vectree/utils.py::read_ply_data()`、`write_ply_data()` 强耦合。改 PLY 字段顺序会影响 VecTree 切片逻辑。

# 9. 源码阅读路线图（建议按这个顺序看）

这部分是面向“真正读懂源码”的路线图。LightGaussian 不是一个从零写的新渲染器，而是在 3DGS 的代码骨架上增加压缩能力。因此读代码时不要一上来钻 CUDA kernel，建议按下面顺序逐层深入：

```text
README / scripts
  |
  v
arguments/__init__.py
  |
  v
scene/dataset_readers.py -> utils/camera_utils.py -> scene/cameras.py
  |
  v
scene/__init__.py -> scene/gaussian_model.py
  |
  v
gaussian_renderer/__init__.py
  |
  v
train_densify_prune.py / prune_finetune.py / distill_train.py
  |
  v
prune.py
  |
  v
submodules/compress-diff-gaussian-rasterization
  |
  v
vectree/vectree.py -> vectree/vq.py -> vectree/utils.py
```

读源码时可以把项目拆成四个层次：

| 层次 | 你要回答的问题 | 主要文件 |
|---|---|---|
| 数据层 | 图像、相机、点云如何进入训练？ | `scene/dataset_readers.py`, `utils/camera_utils.py`, `scene/cameras.py` |
| 表示层 | 一个 Gaussian 到底有哪些参数？这些参数如何训练、保存、剪枝？ | `scene/gaussian_model.py` |
| 渲染层 | Python 如何把 Gaussian 和相机交给 CUDA？CUDA 返回什么？ | `gaussian_renderer/__init__.py`, `submodules/...` |
| 压缩层 | LightGaussian 如何比原始 3DGS 更小？ | `prune.py`, `prune_finetune.py`, `distill_train.py`, `vectree/` |

如果你时间有限，优先读下面 8 个函数：

| 优先级 | 函数 | 为什么重要 |
|---:|---|---|
| 1 | `GaussianModel.create_from_pcd()` | 从点云生成初始 Gaussian，是训练起点 |
| 2 | `GaussianModel.training_setup()` | 定义哪些参数被优化，以及各自学习率 |
| 3 | `gaussian_renderer.render()` | Python 到 CUDA rasterizer 的主入口 |
| 4 | `train_densify_prune.training()` | 从头训练、densify、剪枝、保存的完整循环 |
| 5 | `GaussianModel.densify_and_prune()` | 原始 3DGS 自适应增删点机制 |
| 6 | `prune.prune_list()` | LightGaussian 全局重要性统计入口 |
| 7 | `distill_train.training()` | SH 降阶蒸馏的 teacher-student 逻辑 |
| 8 | `Quantization.quantize()` | VecTree/VQ 压缩的核心流程 |

# 10. 关键张量与形状总表

理解这个项目的最快方式之一，是把所有核心 tensor 的形状记住。下面的 `N` 表示 Gaussian 数量，`K=(sh_degree+1)^2` 表示 SH 系数数量。

| 名称 | 位置 | 典型形状 | 物理意义 | 注意点 |
|---|---|---:|---|---|
| `_xyz` | `GaussianModel` | `[N, 3]` | Gaussian 中心，世界坐标 | 直接优化 |
| `_features_dc` | `GaussianModel` | `[N, 1, 3]` | SH DC 项，近似基础颜色 | 学习率较高 |
| `_features_rest` | `GaussianModel` | `[N, K-1, 3]` | SH 高阶项，视角相关颜色 | 学习率是 `feature_lr/20` |
| `_opacity` | `GaussianModel` | `[N, 1]` | inverse-sigmoid 空间的 alpha | 真值用 `sigmoid` |
| `_scaling` | `GaussianModel` | `[N, 3]` | log 空间的椭球三轴尺度 | 真值用 `exp` |
| `_rotation` | `GaussianModel` | `[N, 4]` | 四元数旋转 | 使用前 normalize |
| `max_radii2D` | `GaussianModel` | `[N]` | 每个 Gaussian 最大屏幕投影半径 | densify/prune 用 |
| `xyz_gradient_accum` | `GaussianModel` | `[N, 1]` | 屏幕空间梯度累计 | densify 用 |
| `denom` | `GaussianModel` | `[N, 1]` | 梯度累计次数 | 求平均梯度用 |
| `screenspace_points` | `render()` | `[N, 3]` | 2D 均值梯度占位 | 不是实际 2D 坐标 |
| `radii` | rasterizer 返回 | `[N]` | 屏幕空间半径 | `radii>0` 表示可见 |
| `visibility_filter` | `render()` 返回 | `[N] bool` | 当前视角可见 mask | densify 只更新可见点 |
| `gaussians_count` | `count_render()` 返回 | `[N]` | 每个 Gaussian 参与统计 | 剪枝候选分数 |
| `important_score` | `count_render()` 返回 | `[N]` | 每个 Gaussian 贡献统计 | LightGaussian 核心 |
| `v_list` | `calculate_v_imp_score()` | `[N]` | volume-aware importance | 分数越低越先删 |

SH 维度展开关系：

| `sh_degree` | `K=(degree+1)^2` | `_features_dc` 通道 | `_features_rest` 通道 | PLY 中 `f_rest_*` 数量 |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3 | 0 | 0 |
| 1 | 4 | 3 | 9 | 9 |
| 2 | 9 | 3 | 24 | 24 |
| 3 | 16 | 3 | 45 | 45 |

PLY 属性顺序非常重要：

```text
x, y, z,
nx, ny, nz,
f_dc_0, f_dc_1, f_dc_2,
f_rest_0 ... f_rest_M,
opacity,
scale_0, scale_1, scale_2,
rot_0, rot_1, rot_2, rot_3
```

这个顺序同时被下面几处依赖：

| 依赖点 | 文件 | 如果顺序改错会怎样 |
|---|---|---|
| 保存 PLY | `GaussianModel.save_ply()` | 后续读取字段混乱 |
| 读取 PLY | `GaussianModel.load_ply()` | assert 失败或属性错位 |
| VecTree 读取 | `vectree/utils.py::read_ply_data()` | SH、opacity、scale、rotation 切片错位 |
| VecTree 写回 | `vectree/utils.py::write_ply_data()` | 反量化 PLY 无法正确渲染 |

# 11. 端到端调用链详解

## 从头训练并剪枝

命令通常类似：

```bash
python train_densify_prune.py -s DATASET -m MODEL --iterations 30000
```

调用链：

```text
main
-> ModelParams / OptimizationParams / PipelineParams
-> safe_state()
-> network_gui.init()
-> training(...)
   -> prepare_output_and_logger()
   -> GaussianModel(dataset.sh_degree)
   -> Scene(dataset, gaussians)
      -> readColmapSceneInfo() or readNerfSyntheticInfo()
      -> cameraList_from_camInfos()
      -> GaussianModel.create_from_pcd()
   -> gaussians.training_setup(opt)
   -> for iteration:
      -> network_gui.try_connect()
      -> gaussians.update_learning_rate()
      -> gaussians.oneupSHdegree()
      -> sample train camera
      -> render()
      -> loss = L1 + DSSIM
      -> loss.backward()
      -> gaussians.add_densification_stats()
      -> gaussians.densify_and_prune()
      -> prune_list() at prune_iterations
      -> calculate_v_imp_score()
      -> gaussians.prune_gaussians()
      -> optimizer.step()
      -> scene.save() / torch.save()
```

训练循环中每一轮最关键的状态变化：

| 阶段 | 输入 | 输出 | 是否反向传播 | 备注 |
|---|---|---|---|---|
| 采样相机 | `scene.getTrainCameras()` | `viewpoint_cam` | 否 | 当前 batch 只有一个视角 |
| 渲染 | camera + gaussians | `image`, `radii`, `viewspace_points` | 是 | CUDA autograd |
| loss | `image`, `gt_image` | scalar loss | 是 | L1 + DSSIM |
| backward | loss | Gaussian 参数梯度 | 是 | 包括 xyz/SH/opacity/scale/rotation |
| densify 统计 | `viewspace_points.grad` | `xyz_gradient_accum` | 否 | `torch.no_grad()` 中累计 |
| densify/prune | 梯度、opacity、radii | 新的 Gaussian 集合 | 否 | 会改 optimizer state |
| optimizer | 参数梯度 | 更新后的参数 | 否 | AdamW |
| LightGaussian 剪枝 | 全局重要性分数 | 更少的 Gaussian | 否 | 只在指定 iteration 做 |

## 加载已有模型剪枝恢复

命令通常类似：

```bash
python prune_finetune.py -s DATASET -m MODEL --start_pointcloud PATH/point_cloud.ply
```

关键差异：

| 对比项 | `train_densify_prune.py` | `prune_finetune.py` |
|---|---|---|
| 初始化 | 从输入点云初始化 Gaussian | 从 checkpoint 或 point_cloud.ply 初始化 |
| 主要目的 | 训练过程中加入剪枝 | 对已有模型做压缩和恢复 |
| densify | 默认启用原始 3DGS densify | 代码保留入口但默认关闭 |
| 剪枝类型 | 默认 volume-aware | 支持多种 `prune_type` |
| 常见输出 | point_cloud, checkpoint, imp_score | 剪枝后 point_cloud, checkpoint, imp_score |

`prune_type` 分支含义：

| `prune_type` | 使用分数 | 直觉 | 风险 |
|---|---|---|---|
| `important_score` | `imp_list` | 按像素贡献删 | 可能偏向删除大而稀疏的结构 |
| `v_important_score` | `calculate_v_imp_score()` | 像素贡献 + 体积修正 | `v_pow` 需要调 |
| `max_v_important_score` | `imp_list * max(scale)` | 保留长条/大尺度结构 | 可能保留不重要的大点 |
| `count` | `gaussian_list` | 按参与次数 | 不直接衡量颜色贡献 |
| `opacity` | `get_opacity` | 简单 baseline | 可能误删低 alpha 细节 |

## SH 蒸馏

命令通常类似：

```bash
python distill_train.py -s DATASET -m STUDENT_MODEL --teacher_model TEACHER.pth --start_checkpoint STUDENT.pth --new_max_sh 2
```

逻辑：

```text
teacher: 高阶 SH Gaussian，通常质量更好，参数更多
student: 低阶 SH Gaussian，参数更少

for each iteration:
    sample camera
    optional augmented camera
    teacher_image = render(teacher).detach()
    student_image = render(student)
    loss = L1(student_image, teacher_image) + DSSIM
    update student only
```

为什么不是直接拟合 GT 图像：

| 训练目标 | 优点 | 缺点 |
|---|---|---|
| 拟合真实图像 | 直接恢复数据集指标 | 低阶 SH 容量不足时可能不稳定 |
| 拟合 teacher 渲染 | 目标更平滑，保留 teacher 行为 | 质量上限受 teacher 限制 |

冻结开关：

| 参数 | 默认状态 | 影响 |
|---|---|---|
| `--enable_covariance` | 默认不启用 | 不传时冻结 scale/rotation |
| `--enable_opacity` | 默认不启用 | 不传时冻结 opacity |
| `--augmented_view` | 默认不启用 | 启用后加入虚拟扰动视角 |

## VecTree / VQ 量化

命令通常类似：

```bash
python vectree/vectree.py \
  --important_score_npz_path MODEL \
  --input_path MODEL/point_cloud/iteration_40000/point_cloud.ply \
  --save_path MODEL \
  --sh_degree 2 \
  --vq_ratio 0.6
```

核心思想：

```text
所有 Gaussian SH 特征
  |
  |-- 高重要性部分: 不量化，直接 half 保存
  |
  |-- 低重要性部分: 用 VQ codebook 表示，只保存 code index
```

`vq_ratio` 的直觉：

| `vq_ratio` | 进入 VQ 的比例 | 直接保存原始 SH 的比例 | 效果倾向 |
|---:|---:|---:|---|
| 0.3 | 30% | 70% | 质量更稳，压缩较弱 |
| 0.6 | 60% | 40% | 常见折中 |
| 0.9 | 90% | 10% | 压缩强，质量风险高 |

`extreme_saving/` 目录文件解释：

| 文件 | 内容 | 用途 |
|---|---|---|
| `metadata.npz` | 点数、原始维度、codebook 大小、SH 维度 | 反量化时恢复形状 |
| `codebook.npz` | VQ codebook | 用 index 还原低重要性 SH |
| `vq_indexs.npz` | bit-pack 后的 code index | 压缩保存低重要性 SH |
| `non_vq_mask.npz` | 哪些 Gaussian 不量化 | 区分原始 SH 和 VQ SH |
| `non_vq_feats.npz` | 高重要性 Gaussian 的 SH | 质量优先保存 |
| `other_attribute.npz` | opacity + scale + rotation | 几何和 alpha 属性 |
| `xyz.npz` | Gaussian 中心 | 空间位置 |

反量化调用链：

```text
render.py/render_video.py --load_vq
-> Scene(..., load_vq=True)
-> GaussianModel.load_vq(model_path)
-> load_vqgaussian(model_path/extreme_saving)
-> 拼回 dense Gaussian attribute matrix
-> GaussianModel 设置 _xyz/_features_dc/_features_rest/_opacity/_scaling/_rotation
-> render()
```

# 12. 关键机制用伪代码重写

## `render()` 的本质

```python
def render(camera, gaussians, pipe, background):
    screenspace_points = zeros_like(gaussians.xyz, requires_grad=True)

    raster_settings = {
        image_size,
        tanfovx, tanfovy,
        bg,
        viewmatrix,
        projmatrix,
        sh_degree,
        camera_center,
        f_count=False,
    }

    if pipe.compute_cov3D_python:
        cov3D = gaussians.get_covariance()
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    if pipe.convert_SHs_python:
        colors = eval_sh(...)
    else:
        shs = gaussians.get_features

    image, radii = rasterizer(
        means3D=gaussians.xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors,
        opacities=gaussians.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D,
    )

    return image, screenspace_points, radii > 0, radii
```

一句话：`render()` 是把 Python 中的 Gaussian 参数转换成 CUDA rasterizer 所需参数，并保留屏幕空间梯度。

## `count_render()` 的本质

```python
def count_render(camera, gaussians, pipe, background):
    raster_settings.f_count = True
    gaussians_count, important_score, image, radii = rasterizer(...)
    return gaussians_count, important_score, image, radii
```

一句话：`count_render()` 和 `render()` 几乎一样，只是打开了 CUDA 中的统计分支。

## `prune_list()` 的本质

```python
def prune_list(gaussians, scene, pipe, background):
    total_count = zeros(N)
    total_score = zeros(N)

    for camera in scene.train_cameras:
        pkg = count_render(camera, gaussians, pipe, background)
        total_count += pkg.gaussians_count.detach()
        total_score += pkg.important_score.detach()

    return total_count, total_score
```

一句话：它用所有训练视角，而不是单个 batch，统计全局重要性。

## `calculate_v_imp_score()` 的本质

```python
volume = scale_x * scale_y * scale_z
ref = sorted(volume, descending=True)[int(0.9 * N)]
volume_weight = (volume / ref) ** v_pow
score = volume_weight * important_score
```

直觉：

| 项 | 作用 |
|---|---|
| `important_score` | 衡量像素贡献 |
| `volume` | 衡量空间覆盖 |
| `v_pow` | 控制体积修正强度 |

## `prune_gaussians()` 的本质

```python
threshold = percentile(import_score, percent)
delete_mask = import_score <= threshold
prune_points(delete_mask)
```

注意 mask 语义：

| 函数 | mask=True 表示 |
|---|---|
| `prune_gaussians()` 内部生成的 `prune_mask` | 删除 |
| `prune_points(mask)` 的 `mask` | 删除 |
| `_prune_optimizer(mask)` 的 `mask` | 保留 |

这个语义差异是二次开发最容易写错的地方之一。

## `densify_and_clone()` 的本质

```python
selected = grad_norm >= threshold and max_scale <= percent_dense * scene_extent
new_gaussian = copy(selected_gaussian)
append_to_optimizer(new_gaussian)
```

适用对象：小而关键的 Gaussian。

## `densify_and_split()` 的本质

```python
selected = grad >= threshold and max_scale > percent_dense * scene_extent
samples = normal(mean=0, std=selected_scale)
new_xyz = selected_xyz + rotation @ samples
new_scale = selected_scale / (0.8 * N)
append(new_gaussians)
delete(original_selected_gaussians)
```

适用对象：大而关键的 Gaussian。

## `cat_tensors_to_optimizer()` 为什么复杂

普通 PyTorch 训练中，参数数量固定；但 3DGS 训练中 Gaussian 数量会变：

```text
densify: N -> N + M
prune:   N -> N - K
```

AdamW optimizer 内部保存：

```text
param tensor
exp_avg
exp_avg_sq
step
```

如果只改 `self._xyz`，不改 optimizer 的 `exp_avg/exp_avg_sq`，下一次 `optimizer.step()` 就会 shape 不匹配。所以：

| 操作 | 参数 tensor | optimizer state |
|---|---|---|
| densify append | `torch.cat(old, new)` | `torch.cat(old_state, zeros_like(new))` |
| prune delete | `old[valid_mask]` | `old_state[valid_mask]` |
| reset opacity | replace tensor | state 置零 |

# 13. 文件级职责和阅读检查表

| 文件 | 必读程度 | 读完应理解什么 |
|---|---:|---|
| `arguments/__init__.py` | 中 | 参数如何自动注册、如何从 `cfg_args` 合并 |
| `scene/dataset_readers.py` | 高 | COLMAP/Blender 如何变成 `SceneInfo` |
| `utils/camera_utils.py` | 中 | 图像分辨率如何缩放，Camera 如何构造 |
| `scene/cameras.py` | 高 | view/projection matrix 如何预计算 |
| `scene/__init__.py` | 高 | Scene 如何选择数据集、加载 PLY/VQ 或初始化点云 |
| `scene/gaussian_model.py` | 最高 | Gaussian 参数、optimizer、densify、prune、PLY/VQ |
| `gaussian_renderer/__init__.py` | 最高 | Python 到 CUDA rasterizer 的桥接 |
| `train_densify_prune.py` | 高 | 完整训练循环 |
| `prune_finetune.py` | 高 | 加载已有模型剪枝恢复 |
| `prune.py` | 高 | 全局重要性统计和 volume-aware score |
| `distill_train.py` | 高 | SH teacher-student 蒸馏 |
| `vectree/vectree.py` | 高 | 量化流程和保存格式 |
| `vectree/vq.py` | 中 | VQ codebook 的 EMA 更新 |
| `vectree/utils.py` | 中 | extreme_saving 如何反量化 |
| `render.py` | 中 | train/test 渲染输出 |
| `render_video.py` | 中 | 相机轨迹渲染 |

读完每个关键文件后，可以用下面的问题自测：

| 文件 | 自测问题 |
|---|---|
| `gaussian_model.py` | 为什么 `_scaling` 不直接保存正数尺度？ |
| `gaussian_model.py` | `prune_points()` 为什么要修改 optimizer state？ |
| `gaussian_renderer/__init__.py` | `screenspace_points` 为什么是 zeros_like？ |
| `prune.py` | 为什么不能只看一个视角做剪枝？ |
| `distill_train.py` | 为什么 teacher 渲染结果要 `.detach()`？ |
| `vectree.py` | 为什么高重要性 Gaussian 不进入 VQ？ |
| `scene/__init__.py` | `load_vq=True` 和 `load_iteration` 的加载路径有什么区别？ |

# 14. 调试建议和断点位置

如果你想跟一遍从训练到渲染的流程，建议按下面位置打断点或加打印：

| 目标 | 断点位置 | 建议观察 |
|---|---|---|
| 确认数据集类型 | `Scene.__init__()` | 是否走 Colmap/Blender |
| 确认相机数量 | `cameraList_from_camInfos()` 返回后 | train/test cameras 数量 |
| 确认初始点数 | `create_from_pcd()` | `fused_point_cloud.shape` |
| 确认参数组 | `training_setup()` | optimizer param group 名称和 shape |
| 确认渲染输出 | `render()` 返回前 | `rendered_image.shape`, `radii>0` 数量 |
| 确认 loss | training loop | `Ll1`, `loss` |
| 确认 densify | `densify_and_prune()` 前后 | Gaussian 数量变化 |
| 确认剪枝分数 | `prune_list()` 返回后 | `imp_list.min/max/mean` |
| 确认剪枝阈值 | `prune_gaussians()` | percentile threshold |
| 确认 VQ 掩码 | `Quantization.quantize()` | `non_vq_mask.sum()`, `vq_mask.sum()` |

推荐打印代码片段：

```python
print("N:", gaussians.get_xyz.shape[0])
print("opacity:", gaussians.get_opacity.min().item(), gaussians.get_opacity.max().item())
print("scale:", gaussians.get_scaling.min().item(), gaussians.get_scaling.max().item())
print("visible:", visibility_filter.sum().item())
print("radii:", radii.max().item())
```

剪枝前建议打印：

```python
print("count:", gaussian_list.min(), gaussian_list.mean(), gaussian_list.max())
print("imp:", imp_list.min(), imp_list.mean(), imp_list.max())
print("v:", v_list.min(), v_list.mean(), v_list.max())
```

VecTree 前建议打印：

```python
print("all feats:", self.feats_bak.shape)
print("vq feats:", self.feats.shape)
print("non_vq:", self.non_vq_mask.sum())
print("vq:", self.vq_mask.sum())
print("codebook:", self.model_vq._codebook.embed.shape)
```

# 15. 重要参数解释

| 参数 | 所在脚本 | 含义 | 调大影响 | 调小影响 |
|---|---|---|---|---|
| `--iterations` | train/prune/distill | 总训练轮数 | 质量更稳，耗时更长 | 更快但可能欠拟合 |
| `--sh_degree` | common | 最大 SH 阶数 | 表达力更强，模型更大 | 模型更小，视角相关外观变弱 |
| `--lambda_dssim` | optimization | DSSIM loss 权重 | 更重结构一致性 | 更重 L1 像素拟合 |
| `--densify_until_iter` | optimization | densify 截止轮数 | 结构增长更久 | 更早固定结构 |
| `--densify_grad_threshold` | optimization | densify 梯度阈值 | 更少 densify | 更多 densify |
| `--opacity_reset_interval` | optimization | opacity 重置周期 | 重置更少 | 重置更频繁 |
| `--prune_iterations` | prune scripts | 剪枝发生轮数 | 多次剪枝 | 少次剪枝 |
| `--prune_percent` | prune scripts | 每次剪掉比例 | 压缩更强，质量风险更大 | 更保守 |
| `--prune_decay` | prune scripts | 多次剪枝比例衰减 | 后续剪枝更温和 | 每次剪枝更接近 |
| `--v_pow` | prune scripts | 体积修正指数 | 更偏向保留大 Gaussian | 更接近纯 important_score |
| `--new_max_sh` | distill | student SH 阶数 | 质量更好，模型更大 | 模型更小，质量风险更大 |
| `--vq_ratio` | vectree | 进入 VQ 的比例 | 压缩更强 | 质量更稳 |
| `--codebook_size` | vectree | VQ codebook 大小 | 量化误差更小，codebook 更大 | 更小但误差更大 |

# 16. 二次开发清单

## 改剪枝策略

最小改动路径：

```text
1. 在 prune.py 新增 score 计算函数
2. 在 prune_finetune.py 增加 --prune_type 分支
3. 调用 gaussians.prune_gaussians(percent, score)
4. 观察剪枝前后 N、PSNR、SSIM、LPIPS
```

必须保证：

| 检查项 | 原因 |
|---|---|
| score shape 是 `[N]` 或 `[N,1]` | `prune_gaussians()` 按 Gaussian 排序 |
| score 越小越应该删 | 现有阈值逻辑是 `<= percentile` |
| score 在 CUDA 上 | 避免 device mismatch |
| 不需要梯度 | 剪枝统计在 `torch.no_grad()` 中用 |

## 改 rasterizer 统计项

需要同时改：

| 层 | 文件 |
|---|---|
| Python 调用 | `gaussian_renderer/__init__.py` |
| Python autograd wrapper | `diff_gaussian_rasterization/__init__.py` |
| C++ binding | `ext.cpp`, `rasterize_points.cu/h` |
| CUDA 实现 | `cuda_rasterizer/forward.cu`, `rasterizer_impl.cu` |
| 上层使用 | `prune.py`, `prune_finetune.py` |

建议流程：

```text
先让 CUDA 返回一个全 0/全 1 tensor
-> 确认 Python 能收到 shape 正确的 tensor
-> 再实现真实统计逻辑
-> 最后接入 prune score
```

## 新增 Gaussian 属性

必须同步维护：

| 位置 | 需要做什么 |
|---|---|
| `GaussianModel.__init__()` | 新增空 tensor |
| `create_from_pcd()` / `load_ply()` | 初始化或加载 |
| `training_setup()` | 加入 optimizer param group |
| `capture()` / `restore()` | checkpoint 保存恢复 |
| `save_ply()` / `construct_list_of_attributes()` | PLY 保存 |
| `_prune_optimizer()` | 删除 Gaussian 时同步裁剪 |
| `cat_tensors_to_optimizer()` | densify 时同步拼接 |
| `gaussian_renderer.render()` | 传给 rasterizer |
| CUDA rasterizer | 真正使用这个属性 |

## 改 VecTree 保存格式

必须同步维护：

| 保存端 | 加载端 |
|---|---|
| `Quantization.fully_vq_reformat()` | `vectree/utils.py::load_vqgaussian()` |
| `write_ply_data()` | `GaussianModel.load_vq()` |

不要只改保存，不改加载；否则 `--load_vq` 会恢复错误属性。

# 17. 一句话总结各核心文件

| 文件 | 一句话 |
|---|---|
| `scene/dataset_readers.py` | 把外部数据集读成统一的 `SceneInfo` |
| `scene/__init__.py` | 把 `SceneInfo` 变成相机列表和 GaussianModel |
| `scene/gaussian_model.py` | 管理所有 Gaussian 参数及其增删、保存、加载 |
| `gaussian_renderer/__init__.py` | 把 Python 参数打包给 CUDA rasterizer |
| `train_densify_prune.py` | 从头训练，并在训练中做 LightGaussian 剪枝 |
| `prune_finetune.py` | 对已有模型剪枝并继续优化恢复质量 |
| `prune.py` | 计算全局重要性分数 |
| `distill_train.py` | 用高阶 teacher 监督低阶 student |
| `vectree/vectree.py` | 把 SH 特征按重要性分成保留和 VQ 两部分 |
| `vectree/vq.py` | 实现 codebook 查表和 EMA 更新 |
| `vectree/utils.py` | 读写量化后的 `extreme_saving` 格式 |
| `render.py` | 批量渲染 train/test 并保存 png |
| `render_video.py` | 沿轨迹渲染视频帧 |

# 18. 建议的阅读节奏

如果你是第一次读这个项目，建议分三轮。

第一轮只看 Python 主链路：

```text
scene/__init__.py
scene/gaussian_model.py
gaussian_renderer/__init__.py
train_densify_prune.py
prune.py
```

目标：知道数据怎么来、Gaussian 怎么表示、训练怎么跑、剪枝怎么删。

第二轮看压缩链路：

```text
prune_finetune.py
distill_train.py
vectree/vectree.py
vectree/vq.py
vectree/utils.py
```

目标：理解 LightGaussian 的 3 个压缩阶段：剪枝、SH 蒸馏、VQ。

第三轮再看 CUDA：

```text
submodules/compress-diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
submodules/compress-diff-gaussian-rasterization/rasterize_points.cu
submodules/compress-diff-gaussian-rasterization/cuda_rasterizer/forward.cu
submodules/compress-diff-gaussian-rasterization/cuda_rasterizer/backward.cu
submodules/compress-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
```

目标：理解 tile-based splatting、排序、alpha blending、反传，以及 LightGaussian 增加的统计分支。
