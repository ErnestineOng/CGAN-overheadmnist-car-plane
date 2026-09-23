# 🧠 Conditional GAN for Car & Plane Image Generation

A Deep Learning project that implements a **Conditional Generative Adversarial Network (CGAN)** for generating grayscale aerial images of **cars and planes** using the **Overhead-MNIST** dataset. The project compares a baseline CGAN with a modified convolutional CGAN and evaluates the generated image distribution using **Fréchet Inception Distance (FID)**.

## 🎯 Project Overview

The objective is to build a GAN-based image generation model that can generate images based on a given class label.

The project consists of two main experiments:

* **Baseline CGAN** using Fully Connected layers
* **Modified CGAN** using Convolutional and Transposed Convolutional layers

The generated images are evaluated using **FID** to measure the similarity between the generated and real image distributions.

## 📊 Dataset

The project uses two classes from the **Overhead-MNIST** dataset:

* **Plane** — 1,000 images
* **Car** — 1,000 images
* **Total:** 2,000 images
* **Image format:** Grayscale
* **Original image size:** 28 × 28 pixels

The dataset is divided into:

* **80% training:** 1,600 images
* **20% testing:** 400 images

The images are resized and normalized before being used for training.

## 🔧 Methodology

### Baseline CGAN

The baseline model uses a Fully Connected architecture.

**Generator**

* Random noise with dimension 100
* Class label embedding
* Fully Connected layers: 128 → 256 → 512 → 1024 → 784
* Batch Normalization
* LeakyReLU activation
* Tanh output activation

**Discriminator**

* Image and class label as inputs
* Fully Connected layers: 512 → 1024 → 1024 → 512 → 1
* LeakyReLU activation
* Sigmoid output

**Training configuration**

* Batch size: 64
* Learning rate: 0.0002
* Optimizer: Adam
* Epochs: 100
* Loss function: Binary Cross Entropy (BCELoss)

### Modified CGAN

The baseline architecture was modified to better handle spatial information in image data.

The main modifications include:

* Image resolution increased from **28 × 28 to 32 × 32**
* Replaced Fully Connected image generation with **ConvTranspose2D**
* Used **Convolutional layers** in the Discriminator
* Added **Spectral Normalization**
* Reduced learning rates to **1e-4 for Generator** and **5e-5 for Discriminator**
* Increased training from **100 to 200 epochs**
* Applied **n_critic = 2**
* Added **label smoothing**
* Added learning rate scheduling using **MultiStepLR**

These modifications were intended to improve training stability and allow the model to better capture spatial features.

## 📈 Results

The models were evaluated using **Fréchet Inception Distance (FID)** on the test data.

| Model         |    FID Score |
| ------------- | -----------: |
| Baseline CGAN | **239.8994** |
| Modified CGAN | **255.2191** |

A lower FID indicates a closer distribution between generated and real images.

The modified CGAN produced a **higher FID score** than the baseline, meaning that the architectural and training modifications did not improve the FID result in this experiment.

## 💡 Key Findings

* The baseline CGAN achieved an FID score of **239.8994**.
* The modified CGAN achieved an FID score of **255.2191**.
* Although the modified model showed **more stable training loss**, this did not translate into a better FID score.
* The convolutional architecture introduced better spatial feature processing, but the limited dataset size may have restricted its effectiveness.
* Lower learning rates and additional stabilization techniques may have made training more stable while also slowing the Generator's learning process.
* The results show that **more complex architecture and more stable training do not necessarily lead to better FID performance**.

## 🛠️ Tools & Technologies

* Python
* PyTorch
* Torchvision
* PyTorch-FID
* NumPy
* Pandas
* Matplotlib
* PIL
* Jupyter Notebook

## 📚 Data Source

The dataset used in this project is **Overhead-MNIST**, obtained from Kaggle.

* Kaggle: Overhead-MNIST
* Paper: *Overhead-MNIST: A Large-Scale Benchmark Dataset for Overhead Imagery*

This project was completed as part of the **Deep Learning Final Examination**.
