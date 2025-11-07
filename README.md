# Probabilistic Machine Learning

A comprehensive implementation and exploration of probabilistic machine learning methods, including diffusion models, Gaussian processes, variational autoencoders, and Bayesian inference techniques.

## 📋 Overview

This repository contains implementations and experiments for key concepts in probabilistic machine learning, developed as part of advanced coursework. The project covers both theoretical derivations and practical implementations of modern probabilistic methods.

## 🎯 Main Components

### 1. Diffusion-Based Generative Models
Implementation and comparison of two guidance strategies for conditional image generation:

- **Classifier-Guided Diffusion (CGD)**: Integrates class-specific information via a trained classifier
- **Classifier-Free Guided Diffusion (CFG)**: Incorporates guidance directly into the diffusion model without a separate classifier

**Key Results:**
- Evaluated on MNIST using Fréchet Inception Distance (FID) and Inception Score (IS)
- CFG generally outperforms CGD, consistent with recent literature
- Best FID: 18.63 (CFG with w=0.8)
- Best IS: 9.99 (both methods with w=2.0-3.0)

### 2. Gaussian Process Regression with Constraints

#### Function Fitting with Constraints
- Implementation of Gaussian Process models with composite kernels (Periodic + RBF)
- Comparison of Maximum A Posteriori (MAP) vs. MCMC approaches
- Analysis of 20 synthetic datasets with cyclic patterns and trends

**Key Features:**
- Automatic kernel structure selection based on data characteristics
- Grid search over hyperparameter space (7,776 configurations)
- Uncertainty quantification with predictive distributions
- Convergence diagnostics (Gelman-Rubin statistics)

#### Learning with Integral Constraints
- Derivation and implementation of GPs with integral constraints
- Trapezoidal rule approximation for integral computation
- Visualization of constrained posterior distributions
- Demonstrates improved alignment with ground truth when constraints match data properties

### 3. Variational Autoencoders (VAE)
PyTorch implementation of a convolutional VAE for MNIST:

**Architecture:**
- Encoder: 28×28 → 14×14 → 7×7 (convolutional downsampling)
- Latent space: 2D (for visualization)
- Decoder: 7×7 → 14×14 → 28×28 (transposed convolutions)

**Training Details:**
- 50 epochs with Adam optimizer (lr=3e-4)
- KL divergence weighted by 0.001 to prevent mode collapse
- Achieves smooth latent space organization with semantically similar digits clustered together

### 4. Bayesian Inference Exercises
Theoretical derivations and implementations including:
- Law of total expectation and variance decomposition
- Markov blankets and d-separation in probabilistic graphical models
- Bayesian networks and conditional independence
- Gaussian mixture models and marginal distributions
- KL divergence minimization for variational inference

## 🛠️ Requirements

```bash
torch
numpy
matplotlib
scipy
tqdm
torchvision
```

## 📊 Results

### Diffusion Models
- Comprehensive FID/IS analysis across guidance scales
- Samples demonstrate enhanced class-specific fidelity
- Trade-off analysis between sample quality and diversity

### Gaussian Processes
- MAP approach achieves superior posterior likelihood compared to MCMC
- MCMC struggles with noise parameter estimation on limited data (20 training points)
- Integral constraints significantly improve predictions with sparse observations

### VAE
- Latent space exhibits expected Gaussian structure
- Clear semantic organization: similar digits cluster together
- Smooth interpolation between digit classes in latent space

## 📁 Repository Structure

```
├── images/               # Generated plots and visualizations
├── imgs/                 # Additional figures
├── implementations/      # Core algorithm implementations
│   ├── diffusion/       # Diffusion model code
│   ├── gp/              # Gaussian process implementations
│   └── vae/             # VAE PyTorch implementation
├── exercises/           # Theoretical derivations
└── README.md
```

## 🚀 Usage

### Running the VAE
```python
python vae_mnist.py
```

### Gaussian Process Experiments
```python
python gp_function_fitting.py
```

### Diffusion Model Training
```python
python train_diffusion.py --guidance {cfg|cgd} --weight 0.8
```

## 📖 Theoretical Background

The project implements concepts from:
- **Bishop's Pattern Recognition and Machine Learning**: Exercises 2.8, 8.9, 8.11, 9.10, 10.4
- **Ho et al. (2020)**: Denoising Diffusion Probabilistic Models
- **Dhariwal & Nichol (2021)**: Diffusion Models Beat GANs
- **Ho & Salimans (2022)**: Classifier-Free Diffusion Guidance

## 📈 Key Findings

1. **CFG vs CGD**: Classifier-free guidance simplifies training and offers better robustness to adversarial attacks, though at the cost of slower inference
2. **GP Inference**: MAP estimation outperforms MCMC on small datasets due to convergence issues with noise parameters
3. **Integral Constraints**: Incorporating global constraints dramatically improves GP predictions when constraints match true data properties
4. **VAE Organization**: 2D latent spaces naturally organize semantically, with interpolation revealing smooth transitions between digit classes

## 👤 Author

Christian Jensen

## 📝 License

MIT License

---

*This project was developed as part of advanced coursework in Probabilistic Machine Learning, combining theoretical understanding with practical implementation of state-of-the-art probabilistic methods.*
