# Deepfake Detection in Medical Images Using VAE-GAN with PSO

##  Overview
This project implements a **Variational Autoencoder – Generative Adversarial Network (VAE-GAN)** framework with **Particle Swarm Optimization (PSO)** for detecting manipulated (deepfake) medical images.

- **VAE**: Encodes and reconstructs medical images, ensuring feature consistency.
- **GAN Discriminator**: Learns to distinguish real from fake images for improved detection.
- **Classification Head**: Predicts real vs fake labels directly from latent space.
- **PSO**: Optimizes hyperparameters such as learning rate, latent size, and convolutional filters.

The framework achieves **high detection accuracy** on medical deepfake datasets, outperforming CNN baselines.

---

## Features
- Medical image preprocessing (noise reduction, contrast enhancement)
- VAE for latent feature extraction & reconstruction
- GAN adversarial learning for robust detection
- Classification head for supervised training
- PSO-based hyperparameter optimization
- Detailed evaluation (Confusion Matrix, Precision, Recall, Specificity, F1-score)

---

## Dataset
You can use datasets such as:
- **320K DeepFake KOA Images** (synthetic X-rays)
- **KNN Model Validation** & **Survey Items**
- Custom medical deepfake datasets

Place datasets inside the `data/` folder.

---

##  Installation
```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-medical-detection.git
cd deepfake-medical-detection

# Install dependencies
pip install -r requirements.txt
