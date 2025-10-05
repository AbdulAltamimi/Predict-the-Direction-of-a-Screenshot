# 🖼️ Screenshot Orientation Classification 
<img width="643" height="328" alt="image" src="https://github.com/user-attachments/assets/f85b05c8-ec12-4bb2-b052-3bd18c9ed29d" />


📱 Predicting Image Rotation using ConvNeXt + GroupKFold + Black Bar Removal

---

## 📌 Project Summary

This notebook presents a complete pipeline for **predicting the orientation (0°, 90°, 180°, 270°)** of screenshot images using **ConvNeXt-Small** pretrained on ImageNet.

The model is trained using:

- Group-aware K-Fold cross-validation (`GroupKFold`)
- CosineAnnealingLR scheduler
- Mixed precision training (AMP)
- Black bar removal (preprocessing)
- Random augmentations (flips + 90° rotations)
- Ensemble inference using best checkpoints

---

## 🔧 Configuration

| Setting           | Value                         |
|------------------|-------------------------------|
| Backbone         | ConvNeXt-Small (pretrained)   |
| Image Size       | 224 × 224                     |
| Batch Size       | 32                            |
| Learning Rate    | 3e-4                          |
| Weight Decay     | 5e-3                          |
| Epochs           | 10                            |
| Scheduler        | CosineAnnealingLR             |
| Early Stopping   | Patience = 3                  |
| Precision        | Mixed (AMP)                   |
| CV Folds         | 3 (GroupKFold)                |

---

## 🧠 Labels

- 0° → Class 0  
- 90° → Class 1  
- 180° → Class 2  
- 270° → Class 3  

Predictions are stored as one of `[0, 90, 180, 270]`.

---

## 📊 Exploratory Data Analysis (EDA)

- Visual samples from the dataset
- Histogram of rotation distribution
- Manual inspection of black bar presence

---

## 🧹 Preprocessing & Augmentations

### ✅ Black Bar Removal
Images are cropped to remove black borders using grayscale thresholding before resizing.

### 🔁 Augmentations
- Random horizontal flips (with label correction)
- Random 90° rotations (with adjusted labels)
- Normalization to ImageNet stats

---

## 🧾 Dataset Class (`ScreenshotDataset`)

Custom PyTorch dataset handles:
- Reading images
- Optional black bar cropping
- Augmentation logic (train only)
- Label conversion (degrees → class)
- Flexible mode switching (train/val/test)

---

## 🏗️ Model Architecture

```python
model = torchvision.models.convnext_small(weights="IMAGENET1K_V1")
model.classifier[2] = nn.Linear(in_features, 4)
```

## 🧪 Training Strategy

We train the model using **3-Fold GroupKFold Cross-Validation** (grouped by `game_label`) to prevent data leakage between similar games.

### ⚙️ Training Details:

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW`
- **Scheduler**: `CosineAnnealingLR`
- **Precision**: Mixed precision training (`torch.cuda.amp`)
- **Early Stopping**: Patience = 3 epochs
- **Augmentations**: Random horizontal flips, 90° rotations (with label adjustment)
- **Preprocessing**: Optional black bar removal using grayscale thresholding
- **Backbone**: ConvNeXt-Small (pretrained on ImageNet)

---

## 📈 Performance Summary

| Fold | Best Validation Accuracy |
|------|---------------------------|
| 1    | 83.55%                    |
| 2    | 90.12%                    |
| 3    | 89.05%                    |
| **Overall OOF Accuracy** | **~87.57%** 🎯 |

> 📌 *OOF = Out-of-Fold predictions collected from each validation set during cross-validation.*

---

## 📉 Loss Curves

### 🔹 Training Loss Per Fold

<img width="693" height="463" alt="image" src="https://github.com/user-attachments/assets/0717e174-1a2c-461d-bf84-3a56e39c6fab" />


---

### 🔹 Validation Loss Per Fold

<img width="687" height="466" alt="image" src="https://github.com/user-attachments/assets/101277ec-8d44-4881-a580-884f086f84d1" />


---


