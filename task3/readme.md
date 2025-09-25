# CLIP Analysis: Inductive Biases and Robustness

## 📂 Dataset Download

- **CIFAR-10**:  
  Downloaded automatically upon first run.

- **PACS Sketch**:
  - Manually download `pacs_sketch.zip` from the official PACS dataset source.
  - Unzip and place the `pacs_sketch` folder in the root directory of this project.
  - Update the notebook variable:
    ```python
    sketch_dataset_path = "./pacs_sketch/pacs_sketch"
    ```

---

## ▶️ Code Flow in `Task3_CLIP_Analysis.ipynb`

The notebook is designed to be run **sequentially**, with each part corresponding to a step in the analysis.

---

### Part 1: Zero-Shot Classification

**Objective**: Test CLIP’s ability to classify images from unseen datasets.

- **Code Block 1 (CIFAR-10)**:

  - Loads pre-trained **CLIP ViT-B/32** and its preprocessor.
  - Loads **CIFAR-10 test set**.
  - Creates text prompts (e.g., `"a photo of a car"`).
  - Encodes text and images into embeddings.
  - Predicts labels by **cosine similarity**.
  - Prints **zero-shot accuracy**.

- **Code Block 2 (PACS Sketch)**:
  - Loads PACS Sketch dataset from local folder.
  - Creates sketch-specific prompts (e.g., `"a sketch of a dog"`).
  - Runs the same zero-shot classification pipeline.
  - Evaluates **out-of-domain generalization**.

---

### Part 2: Prompt Engineering

**Objective**: Study how different text prompt formulations affect CLIP performance.

- Defines multiple prompt templates (e.g., `"{}"`, `"a drawing of a {}"`).
- Runs **zero-shot evaluation** for each template on CIFAR-10.
- Prints accuracy for each template → **direct comparison**.

---

### Part 3: Image-Text Retrieval

**Objective**: Qualitatively test CLIP’s alignment between image and text embeddings.

- Selects 8 diverse CIFAR-10 images.
- Generates corresponding text prompts.
- Computes embeddings for images + texts.
- Performs:
  - **Text-to-Image Retrieval** → best image per text.
  - **Image-to-Text Retrieval** → best text per image.
- Visualizes matches with **matplotlib**, marking correct matches in **green**.

---

### Part 4: Representation Analysis (t-SNE Visualization)

**Objective**: Compare CLIP vs. supervised ResNet-50 feature spaces.

- Loads **CLIP ViT-B/32** and **ResNet-50 baseline**.
- Creates a mixed dataset (CIFAR-10 photos + PACS sketches).
- Extracts high-dimensional embeddings.
- Runs **t-SNE (scikit-learn)** to reduce to 2D.
- Plots two scatter plots:
  - CLIP feature space.
  - ResNet-50 feature space.
- Points are **colored by class** and **marked by domain** (photo vs. sketch).

---

### Part 5 & 6: Bias and Robustness Tests

#### **Shape vs. Texture Bias**

- Loads a sketch image (e.g., dog).
- Creates prompts testing **shape vs. style**:
  - `"a sketch of a dog"` vs. `"a photo of a dog"`.
- Prints CLIP’s confidence scores to show **bias preference**.

#### **Robustness to Noise**

- Loads a clean CIFAR-10 image.
- Adds strong **Gaussian noise**.
- Performs zero-shot classification on both.
- Displays side-by-side:
  - Original image → prediction + confidence.
  - Noisy image → prediction + confidence.

---
