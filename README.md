# Deep Learning Algorithms Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive collection of deep learning algorithms and architectures including Variational Autoencoders, RNN text generation, adversarial robustness, and optimization methods. Professional implementations with research-grade quality and educational value.

## 🎯 Features

- **Generative Models**: Variational Autoencoders (VAE) framework
- **Sequence Models**: RNN/LSTM text generation systems
- **Adversarial Robustness**: Defense against adversarial attacks
- **Optimization**: Advanced optimization function visualization
- **Modular Design**: Reusable components and clean APIs
- **Research Ready**: Publication-quality implementations

## 🚀 Quick Start

```python
from src.variational_autoencoder import VAE
from src.rnn_text_generator import RNNTextGenerator

# Train a Variational Autoencoder
vae = VAE(input_dim=784, latent_dim=32)
vae.train(train_loader, epochs=100)

# Generate new samples
generated_samples = vae.generate(num_samples=64)

# Text generation with RNN
text_generator = RNNTextGenerator()
text_generator.load_data('data/shakespeare.txt')
text_generator.train(epochs=50)

# Generate new text
generated_text = text_generator.generate("To be or not to be", length=100)
```

## 🧠 Algorithm Collection

### 1. Variational Autoencoder (VAE)
- **Location**: `src/variational_autoencoder/`
- **Features**: Probabilistic encoder-decoder with KL divergence
- **Applications**: Image generation, dimensionality reduction, anomaly detection

### 2. RNN Text Generation
- **Location**: `src/rnn_text_generation/`
- **Features**: Character and word-level text generation
- **Applications**: Creative writing, language modeling, style transfer

### 3. Adversarial Robustness
- **Location**: `src/adversarial_robustness/`
- **Features**: FGSM, PGD attacks and defense mechanisms
- **Applications**: Model security, robustness evaluation

### 4. Optimization Visualization
- **Location**: `src/optimization_functions/`
- **Features**: 2D/3D optimization landscape visualization
- **Applications**: Algorithm comparison, education, research

### 5. Logistic Regression Optimization
- **Location**: `src/logistic_regression_optimization/`
- **Features**: Advanced optimization for logistic regression
- **Applications**: Binary classification, optimization benchmarks

## 📁 Project Structure

```
deep-learning-algorithms-toolkit/
├── src/                              # Source algorithms
│   ├── variational_autoencoder/      # VAE implementation
│   ├── rnn_text_generation/          # Text generation
│   ├── adversarial_robustness/       # Security methods
│   ├── optimization_functions/       # Optimization tools
│   └── logistic_regression_optimization/ # LR optimization
├── examples/                         # Usage examples  
├── tests/                           # Test suite
├── docs/                            # Documentation
├── data/                            # Sample datasets
├── models/                          # Pre-trained models
└── README.md                       # This file
```

## 🔧 Algorithm Details

### Variational Autoencoder
```python
# Configure VAE architecture
vae_config = {
    'input_dim': 784,      # MNIST images
    'hidden_dims': [512, 256],
    'latent_dim': 32,
    'beta': 1.0            # KL divergence weight
}

vae = VAE(**vae_config)

# Training with custom loss
loss_history = vae.train(
    train_loader=mnist_loader,
    epochs=100,
    learning_rate=1e-3,
    beta_schedule='constant'  # or 'annealing'
)
```

### RNN Text Generation
```python
# Character-level text generation
rnn_config = {
    'vocab_size': 128,
    'hidden_size': 512,
    'num_layers': 3,
    'dropout': 0.2,
    'temperature': 0.8
}

generator = RNNTextGenerator(**rnn_config)
generator.train_on_text('data/shakespeare.txt', epochs=50)

# Generate with different creativity levels
conservative_text = generator.generate("Hello", temperature=0.5)
creative_text = generator.generate("Hello", temperature=1.2)
```

### Adversarial Robustness
```python
# Test model robustness
robustness_eval = AdversarialEvaluator(model)

# FGSM attack
fgsm_accuracy = robustness_eval.fgsm_attack(
    test_loader=test_data,
    epsilon=0.3
)

# PGD attack  
pgd_accuracy = robustness_eval.pgd_attack(
    test_loader=test_data,
    epsilon=0.3,
    num_steps=20,
    step_size=0.01
)

# Adversarial training for defense
robust_model = robustness_eval.adversarial_training(
    train_loader=train_data,
    attack_method='pgd',
    epochs=50
)
```

## 📊 Performance Benchmarks

Performance metrics will vary based on your specific dataset, model configuration, and hardware setup. Each algorithm is designed to achieve competitive results when properly tuned for your use case.

## 🎨 Visualization Features

### Latent Space Exploration
```python
# Visualize VAE latent space
vae.plot_latent_space(test_data, save_path='latent_space.png')

# Interpolation between samples
interpolation = vae.interpolate(sample1, sample2, steps=10)
vae.save_interpolation_gif(interpolation, 'interpolation.gif')
```

### Optimization Landscapes
```python
# Visualize optimization functions
optimizer_viz = OptimizationVisualizer()

# 2D landscape
optimizer_viz.plot_2d_function('rosenbrock', range_x=(-2, 2), range_y=(-1, 3))

# Optimization path
path = optimizer_viz.optimize_with_history('rastrigin', method='adam')
optimizer_viz.plot_optimization_path(path)
```

## 🔬 Research Applications

### Academic Research
- Generative modeling research
- Optimization algorithm development  
- Adversarial machine learning studies
- Deep learning education

### Industry Applications
- Content generation systems
- Anomaly detection in manufacturing
- Robust AI for safety-critical systems
- Natural language processing pipelines

## 📚 Educational Value

Perfect for:
- Deep learning courses and tutorials
- Research paper implementations
- Algorithm comparison studies
- Hands-on learning experiences

## 🛠 Advanced Features

### Custom Training Loops
```python
# Custom VAE training with logging
trainer = VAETrainer(model=vae, config=training_config)
trainer.add_callback('tensorboard', log_dir='logs/')
trainer.add_callback('model_checkpoint', save_dir='checkpoints/')

history = trainer.train(
    train_loader=train_data,
    val_loader=val_data,
    epochs=100
)
```

### Hyperparameter Optimization
```python
# Automated hyperparameter search
from src.utils import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    model_class=VAE,
    search_space={
        'latent_dim': [16, 32, 64],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'beta': [0.5, 1.0, 2.0]
    }
)

best_params = optimizer.search(train_data, val_data, trials=50)
```

## 🤝 Contributing

We welcome contributions of new algorithms, improvements, and educational materials!

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 References

Implementation based on seminal papers:
- VAE: Kingma & Welling (2014)
- Adversarial Examples: Goodfellow et al. (2015)
- LSTM: Hochreiter & Schmidhuber (1997)

---

🧠 **Deep Learning Algorithms for Research and Education**