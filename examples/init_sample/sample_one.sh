#!/bin/bash

source /users/lyutao/mace-image/mace-venv/bin/activate

# 手动指定使用的 GPU 设备
export OPENBLAS_NUM_THREADS=64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1  # 如果也用 MKL，就也加上

i=$(basename "$1")       # 提取最后一级目录名作为 i

input_dir="$1"           # 原始输入路径（带路径的目录）

cif_file=$(find "$input_dir" -maxdepth 1 -name "*.cif" | head -n 1)
log_dir=logs

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo "Created log directory: $log_dir"
fi

if [ -n "$cif_file" ]; then
    python -u init_sample.py \
        --cif="$cif_file" \
        --run_md \
        --num_to_geo_opt=100 \
        --md_steps=10000 \
        --run_sample \
        --sort_sample \
        --num_to_sample=100 \
        --modelpath="mace-medium-density-agnesi-stress.model" \
        --sample_test \
        --num_to_sample_test=50 \
        --suffix='0508' \
        --cueq \
        > "$log_dir/sample_$i.out" 2>&1
fi


