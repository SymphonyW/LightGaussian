#!/bin/bash

# 获取当前可用 GPU 的编号：显存占用低于阈值时认为该 GPU 空闲
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# 分布式/网络通信使用的初始端口号，每启动一个任务后递增，避免端口冲突
port=6025

# 需要蒸馏微调的数据集名称列表
declare -a run_args=(
    # "bicycle"
    # "bonsai"
    # "counter"
    "kitchen"
    # "room"
    # "stump"
    # "garden"
    # "train"
    # "truck"
)


# 是否启用增强/伪视角进行蒸馏；如果不传该参数，则默认使用训练视角
declare -a virtue_view_arg=(
  "--augmented_view"
)

# 遍历数据集和视角配置，为每个组合启动一次 distill_train.py
for arg in "${run_args[@]}"; do
  for view in "${virtue_view_arg[@]}"; do
    # 等待直到找到一张显存占用低于阈值的 GPU
    while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available. Starting distill_train.py with dataset '$arg' and options '$view' on port $port"

        # 绑定到选中的 GPU，后台启动蒸馏微调任务，并将日志写入 logs 目录
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python distill_train.py \
            -s "/root/datasets/360_v2/$arg" \
            -m "/root/LightGaussian/output/distill/${arg}_${prune_percent}" \
          --start_checkpoint "/root/models/$arg/point_cloud/iteration_30000/point_cloud.ply" \
          --iteration 40000 \
          --eval \
          --teacher_model "/root/models/kitchen/point_cloud/iteration_30000/point_cloud.ply" \
          --new_max_sh 2 \
          --position_lr_max_steps 40000 \
          --enable_covariance \
          $view \
          --port $port > "logs_distill/${arg}${view}.log" 2>&1 &

        # 为下一个任务递增端口号
        ((port++))
        # 给进程预留初始化和占用显存的时间，避免下一轮误判 GPU 仍为空闲
        sleep 60
        break
      else
        echo "No GPU available at the moment. Retrying in 1 minute."
        sleep 60
      fi
    done
  done
done
# 等待所有后台任务执行结束后再退出脚本
wait
echo "All distill_train.py runs completed."
