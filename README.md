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

all python scripts can be found in the cli folder. 

### MLP training

### DFT-label workflow

For this workflow, you only need copy the folder to your computer and provide a confs folder including all cif files you need to calculate. When you move this workflow to your computer, please ensure process.py works and ordered_job.srun stasifiies the requirement of slurm system in your computer. In order to achieve the best acceleration performance, you should sort the structures by similarity defined by the distance in the SOAP descriptor space. You can do it by the scripts in "MLP training".

### Henry Coefficient Calculation


### Heat adsorption Calculation










