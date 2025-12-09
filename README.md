# On-Policy SFT

This repository contains the official implementation of **On-Policy Supervised Fine-Tuning (On-Policy SFT)**.

---

## 📑 Table of Contents

- Overview  
- Environment Setup  
- Installation  
- Reproduce Experiments  
- Configuration Guide  
- Notes & Troubleshooting  

---

## 📖 Overview

On-Policy SFT is a framework for performing supervised fine-tuning in an on-policy manner, aiming to leverage _on-policy strategy_ to mitigate forgetting during SFT, and to reduce entropy collapse in RLVR by _lowering token-wise gradient variance_.

---

## 🛠 Environment Setup

Clone the repository and create the Conda environment:

```sh
git clone https://github.com/AnhaoZhao-LLMer/On_Policy_SFT.git  
cd On_Policy_SFT/osft  
conda create -n osft python=3.10  
conda activate osft  
```
---

## 📦 Installation

Install all required dependencies:
```sh
pip install -r requirements.txt  
pip install flash_attn==2.7.4.post1 --no-build-isolation  # choose a suitable version for your own machine  
pip install -e . --no-dependencies  
```
---

## 🚀 Reproduce Experiments

⚠️ Before running any experiment scripts, make sure you are in the project root directory: On_Policy_SFT/osft

```sh
# Activate the environment:
conda activate osft  

# Run the example training script:
bash examples/osft_1e7_dsr_tau0s6.sh  
```
---

## ⚙️ Configuration Guide

❗ IMPORTANT: Configuration is REQUIRED before running experiments! You MUST modify the scripts inside the `examples/` directory to fit your own environment.

Key parameters to modify include:

### 1. Model Path
Update the path to your base model.

### 2. Logger
The default logger is `swanlab`.  
If you prefer `wandb` or just console output, modify:
```sh
trainer.logger = ['console', 'swanlab']
```
### 3. Shared Memory
When setting:
```sh
actor_rollout_ref.model.use_shm = True
```
Make sure the model name is a valid absolute path on your machine.

### 4. Other Hyperparameters

For example:

- Rollout number  
- Batch size  
- Learning rate  
- Training steps  

These should all be adjusted according to your hardware and task requirements.

---

## 📝 Notes & Troubleshooting

- Make sure your CUDA, PyTorch, and `flash-attn` versions are compatible.
- If installation of `flash_attn` fails, try installing a version suitable for your CUDA and GPU.
- Always verify model paths are absolute paths when using shared memory.
- For logging issues, switch to console or `wandb` for debugging.
