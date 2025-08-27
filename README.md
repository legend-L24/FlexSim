# FlexSim: A Python Package for simulations of water adsorption within MOFs considering the framework flexibility

## Introduction

This code including three main parts, 

1. the simulations of water adsorption within MOFs, 
2. the workflow for DFT-label.
3. the workflow for MLP training, especially the active learning. 

## Installation

Because the development of machine learing potential is fast, the versions of MACE and pytorch should be update according to the version of MACE you used. And update the calculator object in the code. Here, we use python=3.12, torch=2.5.1 and mace_torch=3.12.0. The detailed installation tutorial is shown below.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install mace_torch 
```

If you have further question, please refer the official document of pytorch and mace.

## Tutorial

All python scripts can be found in the cli folder. The examples to use these scripts can be found in the example folder. The explaination of each parameters can be found by this command below (the same for other scripts)

```bash
python init_sample.py --help
```

### MLP training

#### Example to sample the pristine MOF by the MACE-MP-0b2 model in initialization stage

```bash
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
```

#### Example to sample MOF+H2O by the fine-tuned model and do active learning

```bash
python -u activelearn.py \
        --cif="$cif_file" \
        --autosample_gas \ #For the pristine MOFs, --autosample  
        --totalnumber=14000 \
        --sort_sample \
        --number_of_gas=3 \
        --suffix='0508' \
```

### DFT-label workflow

For this workflow, you only need copy the folder to your computer and provide a confs folder including all cif files you need to calculate. When you move this workflow to your computer, please ensure process.py works and ordered_job.srun stasifiies the requirement of slurm system in your computer. In order to achieve the best acceleration performance, you should sort the structures by similarity defined by the distance in the SOAP descriptor space. You can do it by the scripts in "MLP training".

### Henry Coefficient Calculation

```bash
python -u henry_flex.py 
```

### Heat adsorption Calculation

#### Example to run NVT simulation at 300 Kelvin in order to calculate heat of adsorption. 

```bash
python -u heat.py \
        --cif="$cif_file" \
        --run_md \
        --md_steps=1000000 \
        --modelpath="MACE_run-5555.model" \
        --suffix='0508' \
        --cueq \
        --temperature_list="[300]" \
```







