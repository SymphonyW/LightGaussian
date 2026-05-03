#!/bin/bash

# 获取当前可用 GPU 的编号：显存占用低于阈值时认为该 GPU 空闲
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# 分布式/网络通信使用的初始端口号，每启动一个任务后递增，避免端口冲突
port=6041

# 需要处理的数据集名称列表；取消注释或新增条目即可批量运行多个数据集
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
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
  )


# 剪枝比例、后续剪枝衰减系数、体积重要性权重需要一一对应
declare -a prune_percents=(0.66)
# 后续剪枝的衰减率；例如第二次剪枝比例会按 prune_percent * prune_decay 计算
declare -a prune_decays=(1)
# 体积重要性指数；数值越大，Global significance 中体积项的权重越高
declare -a v_pow=(0.1)

# 剪枝策略类型；默认使用论文中的 Global significance，也可以切换为其他策略做对比实验
declare -a prune_types=(
  "v_important_score"
  # "important_score"
  # "count"
  )


# 检查三个参数数组长度是否一致，避免参数组合错位
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ] || [ "${#prune_percents[@]}" -ne "${#v_pow[@]}" ]; then
  echo "The lengths of prune_percents, prune_decays, and v_pow arrays do not match."
  exit 1
fi

# 遍历数据集、剪枝参数和剪枝策略，为每个组合启动一次 prune_finetune.py
for arg in "${run_args[@]}"; do
  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    for prune_type in "${prune_types[@]}"; do
      # 等待直到找到一张显存占用低于阈值的 GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting prune_finetune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
          
          # 绑定到选中的 GPU，后台启动剪枝微调任务，并将日志写入 logs_prune 目录
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python prune_finetune.py \
            -s "PATH/TO/DATASET/$arg" \
            -m "OUTPUT/PATH/${arg}_${prune_percent}" \
            --eval \
            --port $port \
            --start_checkpoint "PATH/TO/CHECKPOINT/$arg/chkpnt30000.pth" \
            --iteration 35000 \
            --prune_percent $prune_percent \
            --prune_type $prune_type \
            --prune_decay $prune_decay \
            --position_lr_max_steps 35000 \
            --v_pow $vp > "logs_prune/${arg}${prune_percent}prunned.log" 2>&1 &

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
done
# 等待所有后台任务执行结束后再退出脚本
wait
echo "All prune_finetune.py runs completed."
