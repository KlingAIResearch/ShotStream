<p align="center" >
    <img src="figs/shotstream_logo.png"  width="55%" >
</p>

<h1 align="center">Streaming Multi-Shot Video Generation for Interactive Storytelling</h1>


<div align="center">
  <p>
    <a href="https://luo0207.github.io/yawenluo/">Yawen Luo</a><sup>1</sup>
    <a href="https://xiaoyushi97.github.io/">Xiaoyu Shi</a><sup>2,✉</sup>
    <a href="https://zhuang2002.github.io/">Junhao Zhuang</a><sup>1</sup>
    <a href="https://yutian10.github.io/">Yutian Chen</a><sup>1</sup>
    <a href="https://liuquande.github.io/">Quande Liu</a><sup>2</sup>
    <a href="https://xinntao.github.io/">Xintao Wang</a><sup>2</sup>
    <a href="https://magicwpf.github.io/">Pengfei Wan</a><sup>2</sup><br>
    <a href="https://tianfan.info/">Tianfan Xue</a><sup>1,3,✉</sup>
  </p>
  <p>
    <sup>1</sup>MMLab, CUHK &nbsp;&nbsp;
    <sup>2</sup>Kling Team, Kuaishou Technology<br>
    <sup>3</sup>CPII under InnoHK &nbsp;&nbsp;
    <sup>✉</sup>Corresponding author
  </p>
</div>

## 📋 Table of Contents
- [📋 Table of Contents](#-table-of-contents)
- [🔥 Updates](#-updates)
- [📷 Introduction](#-introduction)
- [⚙️ Code: ShotStream + Wan2.1-T2V-1.3B](#️-code-shotstream--wan21-t2v-13b)
  - [Inference](#inference)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Download Checkpoints](#2-download-checkpoints)
    - [3. Run Inference](#3-run-inference)
  - [Training](#training)
    - [Step 1: Bidirectional Next-Shot Teacher Model Training](#step-1-bidirectional-next-shot-teacher-model-training)
    - [Step 2: Causal Student Model Distillation](#step-2-causal-student-model-distillation)
      - [Step 2.1: Causal Adaptation Initialization](#step-21-causal-adaptation-initialization)
        - [Step 2.1.1: Get ODE Pairs from Teacher](#step-211-get-ode-pairs-from-teacher)
        - [Step 2.1.2: Get ODE Pairs CSV](#step-212-get-ode-pairs-csv)
        - [Step 2.1.3: Causal Initialization](#step-213-causal-initialization)
      - [Step 2.2: Two-stage Causal Distillation](#step-22-two-stage-causal-distillation)
        - [Step 2.2.1: Intra-shot Self-forcing Distillation](#step-221-intra-shot-self-forcing-distillation)
        - [Step 2.2.2: Inter-shot Self-forcing Distillation](#step-222-inter-shot-self-forcing-distillation)
- [🌟 Citation](#-citation)
- [🤗 Acknowledgement](#-acknowledgement)

**Note:** This open-source repository is a reference implementation. Please note that the original model utilizes internal data, and the prompts in these demo cases exhibit a distribution gap compared to our original training and inference phases.

## 🔥 Updates
- __[2026.03.27]__: Release the [Training and Inference Code](https://github.com/KlingAIResearch/ShotStream) and the [Checkpoints](https://huggingface.co/KlingTeam/ShotStream).
- __[2026.03.27]__: Release the [Project Page](https://luo0207.github.io/ShotStream/) and the [Arxiv]() version.

## 📷 Introduction
**TL;DR:** We propose ShotStream, a novel **causal multi-shot architecture** that enables **interactive storytelling** and **efficient on-the-fly frame generation**, achieving **16 FPS** on a single NVIDIA GPU.

Please watch more video results in our [Project Page](https://luo0207.github.io/ShotStream/).

<div align="center">
  <video src="https://github.com/user-attachments/assets/1ebb6f3d-b6c1-42ae-8fc1-835f49cae682" controls muted width="50%"></video>
</div>


## ⚙️ Code: ShotStream + Wan2.1-T2V-1.3B

### Inference

#### 1. Environment Setup

Create a conda environment and install dependencies:
```bash
git clone https://github.com/KlingAIResearch/ShotStream.git
cd ShotStream
conda create -n shotstream python=3.10 -y
conda activate shotstream
conda install nvidia/label/cuda-12.4.1::cuda
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
Or directly:
```bash
bash tools/setup/env.sh
```

#### 2. Download Checkpoints

Download the checkpoints of Wan-T2V-1.3B and ShotStream:
```bash
apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B wan_models
git clone https://huggingface.co/KlingTeam/ShotStream ckpts
```
Or directly:
```bash
bash tools/setup/download_ckpt.sh
```

#### 3. Run Inference

Autoregressive 4-step Long Multi-Shot Video Generation:

> **Note:** Due to company policy restrictions, the prompts in these demo cases exhibit a distribution shift compared to those used during our original training and inference phases.

```bash
bash tools/inference/causal_fewsteps.sh
```

### Training

#### Step 1: Bidirectional Next-Shot Teacher Model Training

> **Note:**
> 1. You need to update `MASTER_ADDR` in `tools/train/1_basemodel.sh` with the main node's IP address. For multi-node training, the `NNODES` variable also needs to be modified accordingly.
> 2. The multi-shot video example provided is sourced from a public dataset for demonstration purposes. Its captions differ from those used in our actual training set.

Single node:
```bash
bash tools/train/1_basemodel.sh 0
```

Multi-nodes:
```bash
# Run this command on node 0 (main node)
bash tools/train/1_basemodel.sh 0
# Run this command on node 1 (worker node)
bash tools/train/1_basemodel.sh 1
...
```

#### Step 2: Causal Student Model Distillation

##### Step 2.1: Causal Adaptation Initialization

Following [CausVid](https://arxiv.org/pdf/2412.07772v1), we initialize the causal student with the bidirectional teacher's weights. Training all parameters on 5K teacher ODE solution pairs aligns their trajectories, bridging the architectural gap and stabilizing subsequent distillation.

###### Step 2.1.1: Get ODE Pairs from Teacher
```bash
python Teacher_Ode_Sample.py \
  --ckpt_dir ckpts/bidirectional_teacher.pt \
  --save_dir demo/data/ode_sample \
  --data_csv_path demo/data/sample.csv
```

###### Step 2.1.2: Get ODE Pairs CSV
```bash
python get_ode_csv.py \
    -i demo/data/ode_sample \
    -o demo/data/ode_sample.csv
```

###### Step 2.1.3: Causal Initialization

Single node:
```bash
bash tools/train/2_ode_init.sh 0
```

Multi-nodes:
```bash
# Run this command on node 0 (main node)
bash tools/train/2_ode_init.sh 0
# Run this command on node 1 (worker node)
bash tools/train/2_ode_init.sh 1
...
```

##### Step 2.2: Two-stage Causal Distillation

###### Step 2.2.1: Intra-shot Self-forcing Distillation

Single node:
```bash
bash tools/train/3_dmd.sh 0
```

Multi-nodes:
```bash
# Run this command on node 0 (main node)
bash tools/train/3_dmd.sh 0
# Run this command on node 1 (worker node)
bash tools/train/3_dmd.sh 1
...
```

###### Step 2.2.2: Inter-shot Self-forcing Distillation

Single node:
```bash
bash tools/train/4_dmd_long.sh 0
```

Multi-nodes:
```bash
# Run this command on node 0 (main node)
bash tools/train/4_dmd_long.sh 0
# Run this command on node 1 (worker node)
bash tools/train/4_dmd_long.sh 1
...
```
## 🌟 Citation
Please leave us a star 🌟 and cite our paper if you find our work helpful.

```

```

## 🤗 Acknowledgement
- [CausalVid](https://github.com/tianweiy/CausVid): the distillation procedure we built upon. Thanks for their wonderful work.
- [Self Forcing](https://github.com/guandeh17/Self-Forcing): the distillation procedure we built upon. Thanks for their wonderful work.
- [LongLive](https://github.com/NVlabs/LongLive): the distillation procedure we built upon. Thanks for their wonderful work.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon. Thanks for their wonderful work.


