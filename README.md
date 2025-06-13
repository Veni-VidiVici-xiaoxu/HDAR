# HDAR: Human Dimension Attention Regressor for Monocular Occluded Human Mesh Reconstruction

> **Enhancing Monocular Occluded Human Mesh Reconstruction via Human Dimension Attention Mechanism**
>  Yukun Dong, Menghua Wang, Junqi Sun, Long Cheng
>  *Submitted to The Visual Computer*

HDAR introduces a human dimension attention mechanism and a hybrid optimization strategy for improving 3D human body mesh reconstruction under occlusions, based on the PARE framework.

## Highlights

- ⚙️ Improved occlusion robustness via **human dimension attention**.
- 🔁 Iterative optimization combining 2D keypoint ratios and body size constraints.
- 📦 Compatible with SMPL model and 3DPW dataset.
- 📈 Achieves better accuracy than PARE on occluded human benchmarks (e.g., 3DPW-OCC).

## Getting Started

### 1. Clone the Repository

```
bashCopyEditgit clone https://github.com/yourname/HDAR.git
cd HDAR
```

### 2. Install Dependencies

```
bashCopyEdit# pip
source scripts/install_pip.sh

# or conda
source scripts/install_conda.sh
```

### 3. Prepare Data

Download pretrained models and SMPL data (~1.5 GB):

```
bash
source scripts/prepare_data.sh
```

## Running the Demo

```
bash
python scripts/demo.py --image_folder data/demo_images --output_folder logs/demo
```

Outputs will be saved to `logs/demo/hdar_output.pkl` and rendered frames in `logs/demo/`.

## Evaluation

Place evaluation datasets (e.g., 3DPW) under `data/dataset_folders/`, then run:

```
bash

python scripts/eval.py --cfg data/hdar/checkpoints/hdar_config.yaml --opts DATASET.VAL_DS 3dpw
```

## Results on 3DPW-OCC

| Method          | MPJPE ↓  | PA-MPJPE ↓ | PVE ↓    |
| --------------- | -------- | ---------- | -------- |
| PARE            | 90.5     | 56.6       | 107.9    |
| **HDAR (Ours)** | **78.6** | **49.4**   | **94.5** |

## Data & Code Availability

Full code, pretrained models, and evaluation scripts are available at:
 🔗 https://github.com/yourname/HDAR

## Citation

## License

This code is released for non-commercial academic use only. See [LICENSE](LICENSE) for details.
