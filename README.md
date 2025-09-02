# RLForkliftLab

**RLForkliftLab** is an **Isaac Lab / Isaac Sim**–based **reinforcement learning (RL)** environment for **forklifts**, with example **ROS 2 Humble** integration.  
Keywords: RLForkliftLab, RL Forklift Lab, Isaac Lab, Isaac Sim, ROS 2, forklift, PPO, SAC, navigation.

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-22.04-important">
  <img alt="ROS 2" src="https://img.shields.io/badge/ROS%202-Humble-blueviolet">
  <img alt="GPU" src="https://img.shields.io/badge/NVIDIA-GPU%20required-lightgrey">
</p>

## Features
- Forklift kinematics/steering model and Isaac Lab **task** template
- **ROS 2** interface (e.g., `/cmd_vel`) and Docker-based runtime environment
- Example **training/eval** scripts and a custom **env** (PPO/SAC-ready)
- Quick demo launcher (`isaaclab.sh -p ...`)

---

## Prerequisites
- **Ubuntu 22.04**, **ROS 2 Humble**
- **Docker**, **Docker Compose v2**
- **NVIDIA Container Toolkit** (GPU required)
- (Recommended) **NVIDIA Driver 535+**

> When using GPU with GUI, you may need to configure X permissions/display settings.

---

## Installation

```bash
# 1) Clone
git clone https://github.com/jiyuninha/RLForkliftLab.git
cd RLForkliftLab

# 2) (Optional) If the project uses submodules
# git submodule update --init --recursive

# 3) Go to the Docker directory
cd docker

# 4) Build & start (in the background)
docker compose up forklift-lab-base -d --build --remove-orphans
docker compose up forklift-lab-ros2 -d --build --remove-orphans
