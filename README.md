# T²-Nav: Algebraic-Topology Aware Temporal Graph Memory and Loop Detection for Zero-Shot Visual Navigation  

This repository contains the **T²-Nav** implementation, built on top of [UniGoal](https://github.com/bagh2178/UniGoal).  
Our method introduces two new modules and a wrapped Agent class to extend UniGoal for **zero-shot visual navigation with topology-aware temporal memory and loop detection**.  

This codebase is released as part of our ICRA 2026 submission:  
**T²-Nav: Algebraic-Topology Aware Temporal Graph Memory and Loop Detection for Zero-Shot Visual Navigation**.  

---

## Installation  

1. Install dependencies:
  ```bash
  conda create -n T2Nav python=3.8
  conda activate T2Nav
  pip install -r requirements.txt
  ```

2. Clone the UniGoal baseline repository:  
   ```bash
   git clone https://github.com/bagh2178/UniGoal.git
   cd UniGoal
   ```

📂 Dataset Preparation
The folder structure should follow:
   ```bash
  .../
  └── data/
      ├── datasets/
      │   └── instance_imagenav/
      │       └── hm3d/
      │           └── v3/
      │               └── val/
      │                   ├── content/
      │                   │   ├── 4ok3usBNeis.json.gz
      │                   │   ├── 5cdEh9F2hJL.json.gz
      │                   │   └── ...
      │                   └── val.json.gz
      └── scene_datasets/
          └── hm3d_v0.2/
              └── val/
                  ├── 00800-TEEsavR23oF/
                  │   ├── TEEsavR23oF.basis.glb
                  │   └── TEEsavR23oF.basis.navmesh
                  ├── 00801-HaxA7YrQdEC/
                  └── ...
                  └── 00899-58NLZxWBSpk/
  ```
