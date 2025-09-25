# Inductive Biases in CNNs vs. Vision Transformers

### 1. Data Preparation (Scripts 01 & 02)

#### **01_semantic_bias_setup.ipynb**
- **Purpose**: Generate datasets to test color, shape, and texture biases.
- **Actions**:
  - Loads CIFAR-10.
  - Creates **Grayscale CIFAR-10** for color bias testing.
  - Implements **AdaIN style transfer** using pre-trained VGG encoder/decoder (`decoder.pth`, `vgg_normalised.pth`).
  - Generates **Cue-Conflict dataset** by applying texture of one class to the shape of another.
- **Output**:  
  - `test_loader_clean`  
  - `test_loader_grayscale`  
  - `test_loader_cue_conflict`

#### **02_locality_bias_setup.ipynb**
- **Purpose**: Generate datasets to test spatial reasoning and architectural biases.
- **Actions**:
  - **Translated Dataset** via `RandomAffine`.
  - **Occluded Dataset** via `RandomErasing`.
  - **Patch-Shuffled Dataset** via a custom `PatchShuffler` transform.
- **Output**:  
  - `test_loader_translated`  
  - `test_loader_occluded`  
  - `test_loader_shuffled`

---

### 2. Model Fine-Tuning (Script 03)

#### **03_model_finetuning.ipynb**
- **Purpose**: Fine-tune models on CIFAR-10 as a baseline.
- **Actions**:
  - Loads **ResNet-50 (ImageNet pre-trained)** from `torchvision`.
  - Loads **ViT-Base/16** from Hugging Face `transformers`.
  - Replaces final classification heads with 10 outputs (CIFAR-10 classes).
  - Trains with **AdamW optimizer**.
  - Saves best-performing weights by validation accuracy.
  - **Fix for Windows users**: Sets `num_workers=0` in DataLoader for ViT.
- **Output**:  
  - `resnet50_cifar10_best.pth`  
  - `vit_base_16_cifar10_best.pth`

---

### 3. Bias Evaluation and Analysis (Script 04)

#### **04_bias_evaluation.ipynb**
- **Purpose**: Measure inductive biases of ResNet-50 vs. ViT.
- **Actions**:
  - **Loads Models**: Pre-trained weights from previous step.
  - **In-Distribution Performance**: Evaluates on `test_loader_clean`.
  - **Semantic Bias Tests**:
    - Grayscale test → accuracy drop.
    - Cue-Conflict test → computes **Shape Bias percentage**.
  - **Locality Bias Tests**:
    - Translated test → accuracy drop + prediction consistency.
    - Occluded test.
    - Patch-Shuffled test.
  - **Feature Visualization**:
    - Extracts features via hooks.
    - Runs **t-SNE** on clean vs. cue-conflict data.
    - Produces comparative plots.
- **Output**:
  - Accuracy metrics, shape bias percentages.
  - Final bar chart of results.
  - t-SNE visualizations of feature spaces.

---

### 4. Domain Generalization (Script 05)

#### **05_domain_generalization_pacs.ipynb**
- **Purpose**: Test domain generalization using PACS dataset.
- **Actions**:
  - Loads PACS dataset:
    - Train: **Photo, Art, Cartoon**
    - Test: **Sketch**
  - Loads **fresh ImageNet pre-trained ResNet-50 & ViT**.
  - Fine-tunes models for **7 classes**.
  - **Training Enhancements for ViT**:
    - Differential learning rates (higher for head, lower for body).
    - **TrivialAugmentWide** augmentation.
    - **Label smoothing** for regularization.
  - Evaluates on **Sketch domain** (OOD).
- **Output**:
  - Final **out-of-distribution (OOD) generalization accuracy** for ResNet-50 and ViT on PACS Sketch.
