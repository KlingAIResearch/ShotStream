<p align="center" >
    <img src="figs/shotstream_logo.png"  width="55%" >
</p>

# <div align="center">Streaming Multi-Shot Video Generation for Interactive Storytelling<div align="center">


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

<p align="center">
  <a href='https://luo0207.github.io/ShotStream/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  &nbsp;
  <a href=""><img src="https://img.shields.io/static/v1?label=Arxiv&message=ShotStream&color=red&logo=arxiv"></a>
  &nbsp;
  <a href='https://huggingface.co/KlingTeam/ShotStream'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-orange'></a>
</p>

**Note:** This open-source repository is a reference implementation. Please note that the original model utilizes internal data, and the prompts in these demo cases exhibit a distribution gap compared to our original training and inference phases.

## 🔥 Updates
- __[2026.03.19]__: Release the [Project Page](https://camclonemaster.github.io/) and the [Arxiv](https://arxiv.org/abs/2506.03140) version.

## 📷 Introduction
**TL;DR:** We propose CamCloneMaster, a novel **causal multi-shot architecture** that enables **interactive storytelling** and **efficient on-the-fly frame generation**, achieving **16 FPS** on a single NVIDIA GPU.

<div align="center">
  <video controls>
    <source src="figs/demo.mp4" type="video/mp4">
    您的浏览器不支持 HTML5 视频标签。
  </video>
</div>

Please watch more video results in our [Project Page](https://luo0207.github.io/ShotStream/)
## ⚙️ Code: ShotStream + Wan2.1-T2V-1.3B (Inference & Training)
### Inference
**1.** **Environment**: Create a conda environment and install dependencies:
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
or directly: 
```bash
bash tools/setup/env.sh
```

**2. Download checkpoints**:
Download the ckpt of Wan-T2V-1.3B and ShotStream
```bash
apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B wan_models
git clone https://huggingface.co/KlingTeam/ShotStream ckpts
```
or directly: 
```bash
bash tools/setup/download_ckpt.sh
```
**3. Autoregressive 4-step Long Multi-Shot Video Generation**
**Note:** Due to company policy restrictions, the prompts in these demo cases exhibit a distribution shift compared to those used during our original training and inference phases.
```bash
bash tools/inference/causal_fewsteps.sh
```


### Training

### Step 1: Bidirectional Next-Shot Teacher Model Training
**Note:** 
1. You need to update `MASTER_ADDR` in [tools/train/1_basemodel.sh]() with the main node's IP address. For multi-node training, the `NNODES` variable also needs to be modified accordingly.

2. The multi-shot video example provided is sourced from a public dataset for demonstration purposes. Its captions differ from those used in our actual training set.

**Single node:** 
```bash
bash tools/train/1_basemodel.sh 0
```

**Multi-nodes:** 
```bash
# Run this command on node 0 (main node)
bash tools/train/1_basemodel.sh 0
# Run this command on node 1 (worker node)
bash tools/train/1_basemodel.sh 1
...
```

### Step 2: Causal Student Model Distillation
**Step 2.1 Causal Adaptation Initialization**: Following [CausVid](https://arxiv.org/pdf/2412.07772v1), we initialize the causal student with the bidirectional teacher's weights. Training all parameters on 5K teacher ODE solution pairs aligns their trajectories, bridging the architectural gap and stabilizing subsequent distillation.

**Step 2.1.1 Get ODE Pairs from Teacher**
```bash
python Teacher_Ode_Sample.py \
  --ckpt_dir ckpts/bidirectional_teacher.pt \
  --save_dir demo/data/ode_sample \
  --data_csv_path demo/data/sample.csv
```
**Step 2.1.2 Get ODE Pairs CSV**
```python
python /m2v_intern/luoyawen/ECCV2026/ShotStream/get_ode_csv.py \
    -i demo/data/ode_sample \
    -o demo/data/ode_sample.csv
```
**Step 2.1.3 Causal Initialization**

**Single node:** 
```bash
bash tools/train/2_ode_init.sh 0
```

**Multi-nodes:** 
```bash
# Run this command on node 0 (main node)
bash tools/train/2_ode_init.sh 0
# Run this command on node 1 (worker node)
bash tools/train/2_ode_init.sh 1
...
```
**Step 2.2 Two-stage Causal Distillation**:
**Step 2.2.1 Intra-shot Self-forcing Distillation**:
**Single node:** 
```bash
bash tools/train/3_dmd.sh 0
```

**Multi-nodes:** 
```bash
# Run this command on node 0 (main node)
bash tools/train/3_dmd.sh 0
# Run this command on node 1 (worker node)
bash tools/train/3_dmd.sh 1
...
```

**Step 2.2.2 Inter-shot Self-forcing Distillation**:
**Single node:** 
```bash
bash tools/train/4_dmd_long.sh 0
```

**Multi-nodes:** 
```bash
# Run this command on node 0 (main node)
bash tools/train/4_dmd_long.sh 0
# Run this command on node 1 (worker node)
bash tools/train/4_dmd_long.sh 1
...
```