#!/usr/bin/env python3
"""
Logistic Regression Optimization Framework

A comprehensive framework for logistic regression with various optimization algorithms,
including gradient descent, Nesterov momentum, and stochastic gradient descent.

Author: Extracted from ESE5460 HW4
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision as thv
import os
from typing import Tuple, List, Dict, Optional, Callable
from abc import ABC, abstractmethod


class DataProcessor:
    """Data loading and preprocessing utilities."""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        
    def load_mnist(self, classes: List[int] = [0, 1]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess MNIST dataset.
        
        Args:
            classes: List of class labels to include
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print("Dataset not found locally. Downloading...")
            train = thv.datasets.MNIST(self.data_dir, download=True, train=True)
            val = thv.datasets.MNIST(self.data_dir, download=True, train=False)
        else:
            print("Dataset found locally. Loading...")
            train = thv.datasets.MNIST(self.data_dir, download=False, train=True)
            val = thv.datasets.MNIST(self.data_dir, download=False, train=False)
        
        X_train, y_train = self._select_classes(train, classes)
        X_val, y_val = self._select_classes(val, classes)
        
        X_train, y_train = self._normalize_data(X_train, y_train)
        X_val, y_val = self._normalize_data(X_val, y_val)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train + 1)}, Val: {np.bincount(y_val + 1)}")
        
        return X_train, y_train, X_val, y_val
    
    def _select_classes(self, dataset, classes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Select specific classes from dataset."""
        data = dataset.data.numpy()
        targets = dataset.targets.numpy()
        mask = np.isin(targets, classes)
        return data[mask], targets[mask]
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize images and transform labels."""
        # Flatten and normalize images
        X_normalized = X.reshape(X.shape[0], -1) / 255.0
        
        # Transform labels to {-1, +1}
        y_transformed = np.where(y == 0, 1, -1)  # 0 -> +1, 1 -> -1
        
        return X_normalized, y_transformed
    
    def apply_average_pooling(self, X: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """Apply average pooling to reduce image dimensions."""
        pooled_images = []
        
        for image in X:
            # Determine original image dimensions
            side_length = int(np.sqrt(len(image)))
            image_2d = image.reshape((side_length, side_length))
            
            pooled_image = self._average_pool_2d(image_2d, pool_size)
            pooled_images.append(pooled_image.flatten())
        
        return np.array(pooled_images)
    
    def _average_pool_2d(self, image: np.ndarray, pool_size: int) -> np.ndarray:
        """Apply 2D average pooling to single image."""
        height, width = image.shape
        pooled_height = height // pool_size
        pooled_width = width // pool_size
        
        pooled_image = np.zeros((pooled_height, pooled_width))
        
        for i in range(pooled_height):
            for j in range(pooled_width):
                row_start = i * pool_size
                row_end = row_start + pool_size
                col_start = j * pool_size
                col_end = col_start + pool_size
                
                pooled_image[i, j] = np.mean(image[row_start:row_end, col_start:col_end])
        
        return pooled_image
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, num_samples: int = 4) -> None:
        """Visualize sample images."""
        side_length = int(np.sqrt(X.shape[1]))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(2 * num_samples, 2))
        if num_samples == 1:
            axes = [axes]
            
        for i in range(min(num_samples, len(X))):
            image = X[i].reshape((side_length, side_length))
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"Label: {y[i]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


class LogisticRegression:
    """Logistic regression model with various optimization algorithms."""
    
    def __init__(self, regularization: float = 0.0001):
        self.regularization = regularization
        self.w = None
        self.w0 = None
        self.history = {}
        
    def initialize_weights(self, input_dim: int, seed: int = 42) -> None:
        """Initialize model weights."""
        np.random.seed(seed)
        self.w = np.random.randn(input_dim)
        self.w0 = np.random.randn()
        print(f"Initialized weights: w0 = {self.w0:.4f}")
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic regression loss with L2 regularization.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels {-1, +1}
            
        Returns:
            Scalar loss value
        """
        z = self.w0 + np.dot(X, self.w)
        loss = np.mean(np.log(1 + np.exp(-y * z)))
        
        # Add L2 regularization
        reg_term = (self.regularization / 2) * (np.dot(self.w, self.w) + self.w0 ** 2)
        
        return loss + reg_term
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the loss function.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Tuple of (gradient_w, gradient_w0)
        """
        m = X.shape[0]
        z = self.w0 + np.dot(X, self.w)
        t = -y * z
        
        # Sigmoid derivative
        sigmoid = 1 / (1 + np.exp(-t))
        gradient = -y * sigmoid
        
        # Compute gradients with regularization
        dw = (1 / m) * np.dot(X.T, gradient) + self.regularization * self.w
        dw0 = (1 / m) * np.sum(gradient) + self.regularization * self.w0
        
        return dw, dw0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        z = self.w0 + np.dot(X, self.w)
        return np.sign(z)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    @abstractmethod
    def step(self, model: LogisticRegression, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        """Perform one optimization step."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state."""
        pass


class GradientDescent(Optimizer):
    """Standard gradient descent optimizer."""
    
    def step(self, model: LogisticRegression, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        dw, dw0 = model.compute_gradients(X, y)
        model.w -= lr * dw
        model.w0 -= lr * dw0
    
    def reset(self) -> None:
        pass


class NesterovMomentum(Optimizer):
    """Nesterov momentum optimizer."""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.v_w = None
        self.v_w0 = None
    
    def step(self, model: LogisticRegression, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        if self.v_w is None:
            self.v_w = np.zeros_like(model.w)
            self.v_w0 = 0.0
        
        # Nesterov momentum update
        w_lookahead = model.w + self.momentum * self.v_w
        w0_lookahead = model.w0 + self.momentum * self.v_w0
        
        # Create temporary model for gradient computation
        temp_model = LogisticRegression(model.regularization)
        temp_model.w = w_lookahead
        temp_model.w0 = w0_lookahead
        
        dw, dw0 = temp_model.compute_gradients(X, y)
        
        # Update velocities and parameters
        self.v_w = self.momentum * self.v_w - lr * dw
        self.v_w0 = self.momentum * self.v_w0 - lr * dw0
        
        model.w += self.v_w
        model.w0 += self.v_w0
    
    def reset(self) -> None:
        self.v_w = None
        self.v_w0 = None


class StochasticGradientDescent(Optimizer):
    """Stochastic gradient descent optimizer."""
    
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
    
    def step(self, model: LogisticRegression, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for start_idx in range(0, len(X) - self.batch_size + 1, self.batch_size):
            end_idx = start_idx + self.batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            dw, dw0 = model.compute_gradients(X_batch, y_batch)
            model.w -= lr * dw
            model.w0 -= lr * dw0
    
    def reset(self) -> None:
        pass


class OptimizationTrainer:
    """Training manager for logistic regression optimization."""
    
    def __init__(self):
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def train(self,
              model: LogisticRegression,
              optimizer: Optimizer,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              lr: float = 0.005,
              name: str = "training") -> Dict[str, List[float]]:
        """
        Train logistic regression model.
        
        Returns:
            Dictionary containing training metrics
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"Starting training with {name}")
        print(f"Initial loss: {model.compute_loss(X_train, y_train):.6f}")
        
        for epoch in range(epochs):
            # Training step
            optimizer.step(model, X_train, y_train, lr)
            
            # Compute metrics
            train_loss = model.compute_loss(X_train, y_train)
            val_loss = model.compute_loss(X_val, y_val)
            train_acc = model.accuracy(X_train, y_train)
            val_acc = model.accuracy(X_val, y_val)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Acc = {val_acc:.4f}")
        
        model.history[name] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }
        
        return model.history[name]


class ConvergenceAnalyzer:
    """Analyze convergence properties of optimization algorithms."""
    
    def least_squares_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Fit line to data using least squares."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        return slope, intercept
    
    def analyze_convergence_rate(self, losses: List[float], window_size: int = 30) -> Dict[str, float]:
        """
        Analyze convergence rate from loss curve.
        
        Args:
            losses: List of loss values
            window_size: Window for linear fit analysis
            
        Returns:
            Dictionary with convergence metrics
        """
        log_losses = np.log(losses)
        iterations = np.arange(len(log_losses))
        
        # Fit line to initial portion
        if len(log_losses) >= window_size:
            slope, intercept = self.least_squares_fit(
                iterations[:window_size], 
                log_losses[:window_size]
            )
        else:
            slope, intercept = self.least_squares_fit(iterations, log_losses)
        
        # Theoretical convergence rate (inverse of slope magnitude)
        convergence_rate = -1 / slope if slope < 0 else float('inf')
        
        return {
            'slope': slope,
            'intercept': intercept,
            'convergence_rate': convergence_rate,
            'final_loss': losses[-1]
        }


class MomentumAnalyzer:
    """Analyze effect of momentum on optimization."""
    
    def __init__(self, trainer: OptimizationTrainer):
        self.trainer = trainer
        
    def momentum_sweep(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      momentum_values: List[float] = [0.75, 0.8, 0.85, 0.9, 0.95],
                      epochs: int = 100,
                      lr: float = 0.005) -> Dict[float, Dict[str, List[float]]]:
        """
        Test different momentum values.
        
        Returns:
            Dictionary mapping momentum values to training metrics
        """
        results = {}
        
        for momentum in momentum_values:
            print(f"\nTesting momentum = {momentum}")
            
            # Create fresh model and optimizer
            model = LogisticRegression()
            model.initialize_weights(X_train.shape[1])
            optimizer = NesterovMomentum(momentum=momentum)
            
            # Train model
            metrics = self.trainer.train(
                model, optimizer, X_train, y_train, X_val, y_val,
                epochs=epochs, lr=lr, name=f"momentum_{momentum}"
            )
            
            results[momentum] = metrics
        
        return results


def plot_optimization_comparison(results: Dict[str, Dict[str, List[float]]],
                               title: str = "Optimization Comparison") -> None:
    """Plot comparison of different optimization methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    for name, metrics in results.items():
        epochs = range(len(metrics['train_loss']))
        axes[0, 0].plot(epochs, metrics['train_loss'], label=name)
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss comparison
    for name, metrics in results.items():
        epochs = range(len(metrics['val_loss']))
        axes[0, 1].plot(epochs, metrics['val_loss'], label=name)
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss (log scale)
    for name, metrics in results.items():
        epochs = range(len(metrics['train_loss']))
        axes[1, 0].semilogy(epochs, metrics['train_loss'], label=name)
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Training Loss (log scale)')
    axes[1, 0].set_title('Training Loss (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final accuracies
    names = list(results.keys())
    final_accuracies = [results[name]['val_accuracy'][-1] for name in names]
    axes[1, 1].bar(names, final_accuracies)
    axes[1, 1].set_ylabel('Final Validation Accuracy')
    axes[1, 1].set_title('Final Validation Accuracies')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_momentum_analysis(momentum_results: Dict[float, Dict[str, List[float]]]) -> None:
    """Plot momentum analysis results."""
    plt.figure(figsize=(12, 6))
    
    for momentum, metrics in momentum_results.items():
        epochs = range(len(metrics['train_loss']))
        plt.plot(epochs, metrics['train_loss'], label=f"Momentum {momentum}")
    
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Effect of Momentum on Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Demonstrate the logistic regression optimization framework."""
    print("=== Logistic Regression Optimization Framework ===\n")
    
    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    processor = DataProcessor()
    X_train, y_train, X_val, y_val = processor.load_mnist(classes=[0, 1])
    
    # Apply average pooling to reduce dimensionality
    X_train = processor.apply_average_pooling(X_train, pool_size=2)
    X_val = processor.apply_average_pooling(X_val, pool_size=2)
    
    print(f"After pooling - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Visualize samples
    processor.visualize_samples(X_train, y_train, num_samples=4)
    
    # 2. Compare optimization algorithms
    print("\n2. Comparing optimization algorithms...")
    trainer = OptimizationTrainer()
    comparison_results = {}
    
    # Standard Gradient Descent
    model_gd = LogisticRegression()
    model_gd.initialize_weights(X_train.shape[1])
    optimizer_gd = GradientDescent()
    metrics_gd = trainer.train(
        model_gd, optimizer_gd, X_train, y_train, X_val, y_val,
        epochs=100, name="Gradient Descent"
    )
    comparison_results["Gradient Descent"] = metrics_gd
    
    # Nesterov Momentum
    model_nesterov = LogisticRegression()
    model_nesterov.initialize_weights(X_train.shape[1])
    optimizer_nesterov = NesterovMomentum(momentum=0.9)
    metrics_nesterov = trainer.train(
        model_nesterov, optimizer_nesterov, X_train, y_train, X_val, y_val,
        epochs=100, name="Nesterov Momentum"
    )
    comparison_results["Nesterov Momentum"] = metrics_nesterov
    
    # Stochastic Gradient Descent
    model_sgd = LogisticRegression()
    model_sgd.initialize_weights(X_train.shape[1])
    optimizer_sgd = StochasticGradientDescent(batch_size=128)
    metrics_sgd = trainer.train(
        model_sgd, optimizer_sgd, X_train, y_train, X_val, y_val,
        epochs=10, name="SGD"  # Fewer epochs for SGD
    )
    comparison_results["SGD"] = metrics_sgd
    
    plot_optimization_comparison(comparison_results)
    
    # 3. Momentum analysis
    print("\n3. Analyzing momentum effects...")
    momentum_analyzer = MomentumAnalyzer(trainer)
    momentum_results = momentum_analyzer.momentum_sweep(
        X_train, y_train, X_val, y_val,
        momentum_values=[0.75, 0.8, 0.85, 0.9, 0.95],
        epochs=50
    )
    
    plot_momentum_analysis(momentum_results)
    
    # 4. Convergence analysis
    print("\n4. Convergence analysis...")
    analyzer = ConvergenceAnalyzer()
    
    for name, metrics in comparison_results.items():
        if len(metrics['train_loss']) > 30:
            convergence_info = analyzer.analyze_convergence_rate(metrics['train_loss'])
            print(f"\n{name}:")
            print(f"  Slope: {convergence_info['slope']:.6f}")
            print(f"  Convergence Rate: {convergence_info['convergence_rate']:.2f}")
            print(f"  Final Loss: {convergence_info['final_loss']:.6f}")
    
    print("\nFramework demonstration complete!")


if __name__ == "__main__":
    main()