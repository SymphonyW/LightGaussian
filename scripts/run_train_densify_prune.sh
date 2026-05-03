#!/bin/bash

# 获取当前可用 GPU 的编号：显存占用低于阈值时认为该 GPU 空闲
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -v threshold="$mem_threshold" -F', ' '
      $2 < threshold { print $1; exit }
    '
}

# 分布式/网络通信使用的初始端口号，每启动一个任务后递增，避免端口冲突
port=6035

# 需要训练的数据集名称列表；取消注释或新增条目即可批量运行多个数据集
declare -a run_args=(
    "bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    # "train"
    # "truck"
)

# 第一次剪枝使用的剪枝比例
declare -a prune_percents=(0.6)

# 后续剪枝的衰减率
declare -a prune_decays=(0.6)

# 体积重要性指数；会影响基于体积的重要性评分权重
declare -a v_pow=(0.1)

# 剪枝策略类型
declare -a prune_types=(
  "v_important_score"
)

# 检查剪枝比例和衰减率数组长度是否一致，避免参数组合错位
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ]; then
  echo "The number of prune_percents does not match the number of prune_decays."
  exit 1
fi

# 遍历数据集、剪枝参数和剪枝策略，为每个组合启动一次 train_densify_prune.py
for arg in "${run_args[@]}"; do
  # 按相同下标读取 prune_percents、prune_decays 和 v_pow 中的参数
  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    # 遍历不同剪枝策略
    for prune_type in "${prune_types[@]}"; do

      # 等待直到找到一张显存占用低于阈值的 GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting train_densify_prune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"

          # 绑定到选中的 GPU，后台启动训练中的 densify/prune 流程，并将日志写入 logs 目录
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python train_densify_prune.py \
            -s "PATH/TO/DATASET/$arg" \
            -m "OUTPUT/PATH/${arg}" \
            --prune_percent "$prune_percent" \
            --prune_decay "$prune_decay" \
            --prune_iterations 20000 \
            --v_pow "$vp" \
            --eval \
            --port "$port" \
            > "logs/train_${arg}.log" 2>&1 &

          # 使用前需要确保 logs 目录已经存在
          ((port++))

          # 给进程预留初始化和占用显存的时间，避免下一轮误判 GPU 仍为空闲
          sleep 60
          break
        else
          echo "No GPU available at the moment. Retrying in 1 minute."
          sleep 60
        fi
      done

    done  # end for prune_type
  done    # end for i
done      # end for arg

# 等待所有后台任务执行结束后再退出脚本
wait
echo "All train_densify_prune.py runs completed."
