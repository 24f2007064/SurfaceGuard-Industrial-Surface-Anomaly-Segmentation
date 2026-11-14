# SurfaceGuard AI – Industrial Surface Anomaly Segmentation

Multi-class pixel-level anomaly segmentation for industrial surfaces.  
This repository contains a complete training and inference pipeline using a custom **U-Net** trained from scratch in PyTorch.

> Competition-style project focused on localizing and classifying 4 types of surface defects at the pixel level.

---

## Overview

The goal is to detect and segment industrial surface anomalies across **4 defect categories**.  
The model must:
- Localize defects at the pixel level
- Correctly identify the defect type
- Avoid false positives on clean surfaces

The final evaluation metric is a weighted combination of:
- **Foreground Dice Score** (70%)
- **False Positive Score** (20%)
- **Empty Prediction Accuracy** (10%)

---

## Dataset Structure
industrial-surface-defect-detection/

├── public/

│   ├── train/

│   │   ├── 0001.jpg

│   │   ├── 0002.jpg

│   │   └── ...

│   ├── test/

│   │   ├── 0001.jpg

│   │   └── ...

│   ├── train.csv

│   └── sample_submission.csv




### `train.csv` Format
| Column       | Description                          |
|--------------|--------------------------------------|
| image_id     | Unique image identifier              |
| defect_type  | One of: `defect_a`, `defect_b`, `defect_c`, `defect_d` |
| mask_rle     | Run-Length Encoding of the mask (column-major) |

- Each image can have multiple defect types.
- Images without a particular defect have no corresponding row (or empty RLE).

### Image Size
- All images are resized / processed at **128 × 800**

---

## Pipeline Summary

The notebook implements a full end-to-end pipeline:

1. **Data Loading & Preprocessing**
   - RLE decoding (column-major order)
   - Stratified train-validation split (based on presence of any defect)
   - Normalization to `[-1, 1]`

2. **Model**
   - Custom **U-Net** with BatchNorm
   - 4 output channels (one per defect type)
   - Base channels: 32 → 64 → 128 → 256 → 512

3. **Handling Class Imbalance**
   - `WeightedRandomSampler` (rare defects `a` & `b` upweighted)
   - Class-weighted BCE + Dice loss

4. **Training**
   - AdamW + OneCycleLR
   - Mixed precision (AMP)
   - Horizontal + Vertical flip augmentation

5. **Post-processing**
   - Per-class threshold tuning
   - Minimum area filtering
   - Horizontal Flip Test-Time Augmentation (TTA)

6. **Inference**
   - Generates `submission.csv` in the required RLE format

---

## Results

| Approach                    | Approx. Score |
|----------------------------|---------------|
| U-Net from scratch         | ~0.47         |
| Pretrained backbone        | ~0.57         |

*Note: Scores are approximate and based on the competition’s weighted metric.*

---

## How to Run

1. Open the notebook in Kaggle / Colab / local environment with GPU.
2. The notebook automatically downloads the competition data using `kagglehub`.
3. Run all cells sequentially.
4. Final output: `submission.csv`

---

## Key Technical Highlights

- Custom U-Net implementation (not using `segmentation_models_pytorch`)
- Proper RLE encode/decode matching competition format
- Class-imbalance aware training
- Per-class threshold + min-area optimization
- Test-Time Augmentation

---

## Requirements

- Python 3.10+
- PyTorch
- torchvision / PIL
- pandas, numpy, scikit-learn
- kagglehub (for data download)

---

## Notes

- Data is **not** included in this repository due to size and  competition's guideline.
- The notebook is designed to run directly on Kaggle with the competition dataset.
