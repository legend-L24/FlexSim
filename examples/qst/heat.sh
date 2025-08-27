#!/bin/bash

source /users/lyutao/mace-image/mace-venv/bin/activate

i=$(basename "$1")       # 提取最后一级目录名作为 i

input_dir="$1"           # 原始输入路径（带路径的目录）

cif_file=$(find "$input_dir" -maxdepth 1 -name "*.cif" | head -n 1)
log_dir=logs

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    echo "Created log directory: $log_dir"
fi

if [ -n "$cif_file" ]; then
    python -u /users/lyutao/softwares/sampling_code/heat.py \
        --cif="$cif_file" \
        --run_md \
        --md_steps=1000000 \
        --modelpath="/capstor/scratch/cscs/lyutao/modelfiles/mof420/MACE_run-5555.model" \
        --suffix='0508' \
        --cueq \
        --temperature_list="[300]" \
        >> "$log_dir/log_$i.out" 2>&1
fi
