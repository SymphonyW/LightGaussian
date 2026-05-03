#!/bin/bash

# 需要进行 VecTree 量化的场景列表；取消注释下方示例可批量处理多个场景
# SCENES=(bicycle bonsai counter garden kitchen room stump train truck)
SCENES=(room)

# VQ_RATIO 控制参与向量量化的比例，CODEBOOK_SIZE 控制码本大小
VQ_RATIO=0.6
CODEBOOK_SIZE=8192

# 遍历每个场景，构造输入点云、重要性分数和输出目录路径
for SCENE in "${SCENES[@]}"
do
    IMP_PATH=./vectree/pruned_distilled/${SCENE}
    INPUT_PLY_PATH=./vectree/pruned_distilled/${SCENE}/iteration_40000/point_cloud.ply
    SAVE_PATH=./vectree/output/${SCENE}

    # 绑定到 0 号 GPU 运行 VecTree 量化，并把结果保存到 SAVE_PATH
    CMD="CUDA_VISIBLE_DEVICES=0 python vectree/vectree.py \
    --important_score_npz_path ${IMP_PATH} \
    --input_path ${INPUT_PLY_PATH} \
    --save_path ${SAVE_PATH} \
    --vq_ratio ${VQ_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    "
    eval $CMD
done
