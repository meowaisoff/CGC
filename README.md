# Margin Inflation Predicts Calibration Failure in Fine-Tuned Vision-Language Models

Official implementation of **"Margin Inflation Predicts Calibration Failure in Fine-Tuned Vision-Language Models"**.

## Overview

Vision-language models (VLMs) such as CLIP achieve strong downstream performance through lightweight fine-tuning and prompt learning. However, this adaptation often produces poorly calibrated confidence estimates. This work identifies **score-separation dynamics** as the fundamental cause of calibration failure and proposes:

- **Calibration Failure Predictor (CFP)**: A lightweight indicator that captures the relationship between score-separation and score variability, showing strong correlation with Expected Calibration Error (ECE).
- **Confidence-Guided Calibration (CGC)**: A post-hoc framework that adapts its correction strength based on measured score-separation dynamics, reducing ECE by up to 67% while maintaining accuracy.

## Key Contributions

- Comprehensive analysis showing that fine-tuning progressively amplifies separation between the highest-scoring class and competing classes, inflating confidence disproportionately to accuracy gains.
- CFP as a principled predictor of calibration failure without requiring model retraining.
- CGC framework that outperforms temperature scaling, vector scaling, matrix scaling, Platt scaling, DAC, and CAC across diverse settings.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cgc-vlm-calibration.git
cd cgc-vlm-calibration

# Create conda environment
conda create -n cgc python=3.9
conda activate cgc

# Install dependencies
pip install torch torchvision torchaudio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Install DASSL for trainer framework
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch.git
```

## Datasets

Download and prepare the following datasets:

| Dataset | Task | Classes | Split |
|---------|------|---------|-------|
| ImageNet-1K | General object recognition | 1,000 | Base / Val / Test |
| ImageNet-V2 | Distribution shift | 1,000 | Test |
| ImageNet-Sketch | Distribution shift | 1,000 | Test |
| ImageNet-R | Distribution shift | 200 | Test |
| ImageNet-A | Distribution shift | 200 | Test |
| SUN397 | Scene recognition | 397 | Base / Novel |
| StanfordCars | Fine-grained car classification | 196 | Base / Novel |
| Food101 | Food recognition | 101 | Base / Novel |
| Flowers102 | Fine-grained flower classification | 102 | Base / Novel |
| UCF101 | Action recognition | 101 | Base / Novel |

Follow the [CoOp dataset setup](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) for directory structure and preprocessing.

## Methods

### Prompt Learning (In-Distribution)

| Method | Description | Config Key |
|--------|-------------|------------|
| CoOp | Context Optimization with learnable continuous prompts | `TRAINER.COOP` |
| CoCoOp | Conditional Context Optimization with image-conditional prompts | `TRAINER.COCOOP` |
| KgCoOp | Knowledge-guided Context Optimization preserving pre-trained knowledge | `TRAINER.KGCOOP` |

### Adapter-Based (Out-of-Distribution)

| Method | Description |
|--------|-------------|
| CLIP-Adapter | Lightweight feature adapters in vision and text branches |
| Tip-Adapter | Training-free adaptation using cache features |
| TaskRes | Task-specific residual adapters for textual embeddings |

### Calibration Baselines

- Temperature Scaling
- Vector Scaling
- Matrix Scaling
- Platt Scaling
- Distance-Aware Calibration (DAC)
- Contrast-Aware Calibration (CAC)
- **CGC (Ours)**

## Training

### CoOp (16-shot)

```bash
python train.py   --root /path/to/datasets   --seed 1   --trainer CoOp   --dataset-config-file configs/datasets/oxford_pets.yaml   --config-file configs/trainers/CoOp/rn50_16shots.yaml   --output-dir output/coop/oxford_pets
```

### CoCoOp (16-shot)

```bash
python train.py   --root /path/to/datasets   --seed 1   --trainer CoCoOp   --dataset-config-file configs/datasets/oxford_pets.yaml   --config-file configs/trainers/CoCoOp/rn50_16shots.yaml   --output-dir output/cocoop/oxford_pets
```

### KgCoOp (16-shot)

```bash
python train.py   --root /path/to/datasets   --seed 1   --trainer KgCoOp   --dataset-config-file configs/datasets/oxford_pets.yaml   --config-file configs/trainers/KgCoOp/rn50_16shots.yaml   --output-dir output/kgcoop/oxford_pets
```

## Applying CGC Calibration

After training a model, apply CGC post-hoc calibration using validation logits:

```python
import torch
from cgc import ConfidenceGuidedCalibration

# Load validation and test logits
val_logits = torch.load("val_logits.pt")   # [N_val, C]
val_labels = torch.load("val_labels.pt")   # [N_val]
test_logits = torch.load("test_logits.pt") # [N_test, C]

# Initialize CGC
cgc = ConfidenceGuidedCalibration(beta=15.0, delta_0=0.7, n_bins=15)

# Fit on validation set
cgc.fit(val_logits, val_labels)

# Apply to test set
calibrated_probs = cgc.predict_proba(test_logits)
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 15.0 | Sharpness of sigmoid transition |
| `delta_0` | 0.7 | Reference separation threshold |
| `n_bins` | 15 | Number of bins for ECE computation |
| `use_platt` | True | Whether to apply Platt scaling refinement |

## Evaluation

```bash
python eval.py   --root /path/to/datasets   --trainer CoOp   --dataset-config-file configs/datasets/oxford_pets.yaml   --model-dir output/coop/oxford_pets   --eval-only   --apply-cgc
```

## Results

### In-Distribution Few-Shot Learning (ViT-L/14, 16-shot)

| Method | SUN397 ECE | StanfordCars ECE | Food101 ECE | Flowers102 ECE | UCF101 ECE |
|--------|-----------|------------------|-------------|----------------|------------|
| CoOp | 2.84 / 3.98 | 3.11 / 4.35 | 1.93 / 2.71 | 0.86 / 1.20 | 1.89 / 2.65 |
| CoOp+CGC | **1.52** / **2.13** | **1.89** / **2.67** | **1.14** / **1.59** | **0.52** / **0.74** | **1.23** / **1.67** |

*(Base / Novel class ECE shown)*

### Out-of-Distribution Domain Generalization (ViT-L/14)

| Method | ImageNet ECE | ImageNet-V2 ECE | ImageNet-Sketch ECE | ImageNet-R ECE | ImageNet-A ECE |
|--------|-------------|-----------------|---------------------|----------------|----------------|
| TaskRes | 4.50 | 0.70 | 9.50 | 13.50 | 7.05 |
| TaskRes+CGC | **1.90** | **0.60** | **2.50** | **6.50** | **2.88** |

### Calibration Method Comparison (Food101, ViT-B/16)

| Method | ECE (%) | ACC (%) |
|--------|---------|---------|
| Base Model | 8.42 | 89.34 |
| Temperature Scaling | 6.18 | 89.41 |
| Vector Scaling | 5.93 | 89.38 |
| Matrix Scaling | 5.71 | 89.42 |
| Platt Scaling | 5.42 | 89.45 |
| DAC | 4.87 | 89.51 |
| CAC | 4.56 | 89.47 |
| **CGC (Ours)** | **3.21** | **89.62** |





## Acknowledgements

This codebase builds upon:
- [CoOp](https://github.com/KaiyangZhou/CoOp) — Prompt learning framework
- [CLIP](https://github.com/openai/CLIP) — Vision-language pre-training
- [DASSL](https://github.com/KaiyangZhou/Dassl.pytorch) — Domain adaptation and semi-supervised learning

## License

This project is licensed under the MIT License.
