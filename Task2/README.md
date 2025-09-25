# Image Generation with GANs and VAEs on CIFAR-10

This project explores and compares two prominent generative models, a Generative Adversarial Network (GAN) and a Variational Autoencoder (VAE), for the task of image generation on the CIFAR-10 dataset. The goal is to understand their different approaches to learning data distributions and to visually and quantitatively analyze their performance.

## Table of Contents

- [Models](#models)
  - [Generative Adversarial Network (GAN)](#generative-adversarial-network-gan)
  - [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Image Generation](#image-generation)
  - [Image Reconstruction (VAE)](#image-reconstruction-vae)
  - [Latent Space Interpolation](#latent-space-interpolation)
  - [Latent Dimension Variation (VAE)](#latent-dimension-variation-vae)
- [Evaluation](#evaluation)
  - [Quantitative Metrics](#quantitative-metrics)
  - [Out-of-Distribution Detection](#out-of-distribution-detection)
- [Latent Space Analysis](#latent-space-analysis)

## Models

### Generative Adversarial Network (GAN)

- **Why we use it**: GANs are known for producing sharp, high-fidelity images. Their adversarial training process pushes the generator to create increasingly realistic outputs.  
- **How it works**: Two networks are trained in a zero-sum game.  
  - The **Generator** takes a random noise vector and tries to produce a realistic image.  
  - The **Discriminator** is trained to distinguish real images (from the dataset) from the generator's fakes.  
  As training progresses, the generator gets better at fooling the discriminator, resulting in higher-quality images.  

### Variational Autoencoder (VAE)

- **Why we use it**: VAEs excel at learning a smooth and continuous latent representation of the data. This makes them ideal for tasks like interpolation and understanding the underlying factors of variation in images.  
- **How it works**: A VAE consists of two parts that are trained together.  
  - The **Encoder** compresses an input image into a probability distribution (defined by a mean and variance) in a low-dimensional latent space.  
  - The **Decoder** takes a point sampled from this latent distribution and reconstructs the original image.  
  The model is trained to both reconstruct the input accurately and to keep the latent distributions organized, which prevents overfitting and encourages a structured representation.  

## Dataset

The models are trained on the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 classes. This is a standard benchmark for generative models. The dataset is automatically downloaded and preprocessed when running the script.  

For evaluating out-of-distribution detection, the **CIFAR-100 dataset** is also used.  

## Requirements

To run this project, you will need the following libraries:  

- PyTorch  
- Torchvision  
- Matplotlib  
- NumPy  
- scikit-learn  
- torchmetrics  

You can install these dependencies using pip:  

pip install torch torchvision matplotlib numpy scikit-learn torchmetrics  

---

## Usage

The primary script is a Jupyter Notebook containing the implementation, training, and evaluation of both models.  

---

## Image Generation

**Why we do it:**  
This is the primary goal of a generative model: to create new, plausible data from scratch.  

**How we do it:**  
We feed a random vector (noise) into the trained Generator (GAN) or Decoder (VAE). The network transforms this simple input into a complex, high-dimensional image.  

**GAN Image Generation**  
generate_and_plot(generator=Gen, z_dim=100, num_images=25)  

**VAE Image Generation**  
generate_and_plot_vae(generator=vae.decoder, z_dim=64, num_images=25)  

---

## Image Reconstruction (VAE)

**Why we do it:**  
To verify that the VAE has learned a meaningful compression of the data. A good reconstruction proves the latent space captures the image's essential information.  

**How we do it:**  
An image is passed through the encoder to get its latent code, which is then immediately passed to the decoder to see how well it can be rebuilt.  

generate_and_plot_vae_recon(vae, next(iter(train_loader)), num_images=10)  

---

## Latent Space Interpolation

**Why we do it:**  
To visualize the "smoothness" of the latent space. A seamless transition between two different images (e.g., from a car to a truck) shows the model has learned a continuous representation.  

**How we do it:**  
We encode two images to get their latent vectors, z1 and z2. We then generate images from a series of vectors that are weighted averages between z1 and z2, creating a smooth transformation.  

**VAE Interpolation**  
interpolate(img1, img2, vae, 10)  

**GAN Interpolation**  
interpolate_gan(Gen, 100)  

---

## Latent Dimension Variation (VAE)

**Why we do it:**  
To discover what abstract visual features (like orientation, color, or style) each latent dimension controls.  

**How we do it:**  
We encode an image to get its latent vector. We then systematically vary the value of one dimension at a time while keeping others fixed and decode the resulting vectors to see how the image changes.  

visualize_latent_dimension_variation(img_numpy, vae, 10, 9)  

---

## Evaluation

### Quantitative Metrics

**Why we use them:**  
Visual inspection is subjective. Metrics provide an objective, numerical score to reliably compare the performance of different generative models.  

**How they work:**  
- **Inception Score (IS):** Uses a pre-trained image classification model (Inception-v3) to measure image quality (are generated images recognizable?) and diversity (does the model produce a wide variety of classes?). **Higher is better.**  
- **Fréchet Inception Distance (FID):** Compares the statistical properties of features from real and generated images (also extracted using the Inception-v3 model). It measures how similar the two distributions are. **Lower is better.**  

---

## Out-of-Distribution Detection

**Why we do it:**  
To test if the VAE can identify images that are different from its training data. This is a common application for anomaly detection.  

**How we do it:**  
A VAE should be good at reconstructing images it was trained on (in-distribution) but struggle with unfamiliar ones (out-of-distribution). We compare the reconstruction error for images from **CIFAR-10** and **CIFAR-100**. A significantly higher error for CIFAR-100 indicates successful OOD detection.  

---

## Latent Space Analysis

**Why we do it:**  
To visualize the high-dimensional latent space in 2D and see if the model has learned to group similar data. This helps us understand the structure of the learned representations.  

**How we do it:**  
We use **t-SNE**, a dimensionality reduction algorithm, to project the latent vectors of the training data into a 2D scatter plot. We then color each point by its true class label to see if distinct clusters have formed.  
