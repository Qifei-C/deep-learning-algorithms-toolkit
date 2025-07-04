#!/usr/bin/env python3
"""
Variational Autoencoder Framework

A comprehensive framework for training and evaluating Variational Autoencoders (VAEs)
with MNIST data, including ELBO analysis, reconstruction evaluation, and text generation.

Author: Extracted from ESE5460 HW5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Callable
import os


class MNISTDataManager:
    """MNIST data loading and preprocessing utilities."""
    
    def __init__(self, 
                 data_dir: str = './data',
                 image_size: int = 14,
                 binary_threshold: float = 0.5):
        self.data_dir = data_dir
        self.image_size = image_size
        self.binary_threshold = binary_threshold
        
    def get_datasets(self, 
                    num_images_per_class: int = 1000,
                    train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        """
        Create MNIST datasets with subsampling and train/validation split.
        
        Args:
            num_images_per_class: Number of images per digit class
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > self.binary_threshold).float())
        ])
        
        full_dataset = datasets.MNIST(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Subsample dataset
        subsampled_dataset = self._subsample_dataset(full_dataset, num_images_per_class)
        
        # Train/validation split
        train_size = int(train_split * len(subsampled_dataset))
        val_size = len(subsampled_dataset) - train_size
        train_dataset, val_dataset = random_split(subsampled_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
        
        print(f"Dataset sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}")
        print(f"Image shape: {self.image_size}x{self.image_size}")
        
        return train_loader, val_loader
    
    def _subsample_dataset(self, dataset, num_images_per_class: int) -> Subset:
        """Subsample dataset to have equal number of images per class."""
        indices = []
        
        for digit in range(10):
            digit_indices = np.where(dataset.targets.numpy() == digit)[0]
            selected_indices = np.random.choice(
                digit_indices, 
                num_images_per_class, 
                replace=False
            )
            indices.extend(selected_indices)
        
        return Subset(dataset, indices)


class Encoder(nn.Module):
    """VAE Encoder network."""
    
    def __init__(self, input_dim: int = 196, hidden_dim: int = 128, latent_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent mean and log-variance.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mean, log_variance)
        """
        h = torch.tanh(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class Decoder(nn.Module):
    """VAE Decoder network."""
    
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128, output_dim: int = 196):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstruction of shape (batch_size, output_dim)
        """
        h = torch.tanh(self.fc1(z))
        reconstruction = torch.sigmoid(self.fc2(h))
        return reconstruction


class VariationalAutoencoder(nn.Module):
    """Complete VAE model combining encoder and decoder."""
    
    def __init__(self, 
                 input_dim: int = 196,
                 hidden_dim: int = 128,
                 latent_dim: int = 8):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        
        print(f"VAE Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Parameters: {self.count_parameters():,}")
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for backpropagation through sampling.
        
        Args:
            mean: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction1, reconstruction2, mean, logvar)
        """
        # Flatten input
        x_flat = x.view(-1, self.encoder.fc1.in_features)
        
        # Encode
        mean, logvar = self.encoder(x_flat)
        
        # Sample twice for ELBO estimation
        z1 = self.reparameterize(mean, logvar)
        z2 = self.reparameterize(mean, logvar)
        
        # Decode
        x_recon1 = self.decoder(z1)
        x_recon2 = self.decoder(z2)
        
        return x_recon1, x_recon2, mean, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decoder(z)
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through encoder-decoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstruction
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(-1, self.encoder.fc1.in_features)
            mean, logvar = self.encoder(x_flat)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decoder(z)
        return reconstruction
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VAETrainer:
    """Training manager for VAE models."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def compute_elbo_loss(self, 
                         recon1: torch.Tensor,
                         recon2: torch.Tensor,
                         target: torch.Tensor,
                         mean: torch.Tensor,
                         logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO loss components.
        
        Args:
            recon1, recon2: Two reconstruction samples
            target: Original input
            mean, logvar: Latent distribution parameters
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_divergence)
        """
        batch_size = target.size(0)
        target_flat = target.view(-1, recon1.size(-1))
        
        # Reconstruction loss (negative log-likelihood)
        recon_loss1 = F.binary_cross_entropy(recon1, target_flat, reduction='sum') / batch_size
        recon_loss2 = F.binary_cross_entropy(recon2, target_flat, reduction='sum') / batch_size
        avg_recon_loss = (recon_loss1 + recon_loss2) / 2
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
        
        # Total ELBO loss
        total_loss = avg_recon_loss + kl_div
        
        return total_loss, avg_recon_loss, kl_div
    
    def train_model(self,
                   model: VariationalAutoencoder,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 50,
                   lr: float = 0.001) -> Dict[str, List[float]]:
        """
        Train VAE model.
        
        Returns:
            Dictionary containing training metrics
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        train_recon_losses = []
        train_kl_divs = []
        val_log_likelihoods = []
        
        print(f"Training VAE for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_div = 0.0
            
            for images, _ in train_loader:
                images = images.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                recon1, recon2, mean, logvar = model(images)
                
                # Compute loss
                total_loss, recon_loss, kl_div = self.compute_elbo_loss(
                    recon1, recon2, images, mean, logvar
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_div += kl_div.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_kl_div = epoch_kl_div / len(train_loader)
            
            train_losses.append(avg_epoch_loss)
            train_recon_losses.append(-avg_recon_loss)  # Negative for log-likelihood
            train_kl_divs.append(avg_kl_div)
            
            # Validation log-likelihood
            val_log_likelihood = self._compute_validation_log_likelihood(model, val_loader)
            val_log_likelihoods.append(val_log_likelihood)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Loss: {avg_epoch_loss:.4f}')
                print(f'  Reconstruction: {avg_recon_loss:.4f}')
                print(f'  KL Divergence: {avg_kl_div:.4f}')
                print(f'  Val Log-Likelihood: {val_log_likelihood:.4f}')
        
        return {
            'train_loss': train_losses,
            'reconstruction_loss': train_recon_losses,
            'kl_divergence': train_kl_divs,
            'val_log_likelihood': val_log_likelihoods
        }
    
    def _compute_validation_log_likelihood(self, 
                                         model: VariationalAutoencoder,
                                         val_loader: DataLoader) -> float:
        """Compute validation log-likelihood."""
        model.eval()
        log_likelihoods = []
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(self.device)
                recon1, recon2, _, _ = model(images)
                
                images_flat = images.view(-1, recon1.size(-1))
                recon_loss1 = F.binary_cross_entropy(recon1, images_flat, reduction='sum') / images_flat.size(0)
                recon_loss2 = F.binary_cross_entropy(recon2, images_flat, reduction='sum') / images_flat.size(0)
                
                log_likelihood = -(recon_loss1 + recon_loss2) / 2
                log_likelihoods.append(log_likelihood.item())
                break  # Use only one batch for efficiency
        
        return sum(log_likelihoods) / len(log_likelihoods)


class VAEVisualizer:
    """Visualization utilities for VAE analysis."""
    
    def __init__(self, image_size: int = 14):
        self.image_size = image_size
        
    def plot_training_curves(self, metrics: Dict[str, List[float]]) -> None:
        """Plot ELBO components during training."""
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ELBO components
        axes[0].plot(epochs, metrics['reconstruction_loss'], label='Reconstruction Loss', color='blue')
        axes[0].plot(epochs, metrics['kl_divergence'], label='KL Divergence', color='red')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss Component')
        axes[0].set_title('ELBO Components during Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Total loss
        axes[1].plot(epochs, metrics['train_loss'], label='Total Loss', color='green')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Total Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Validation log-likelihood
        val_epochs = range(1, len(metrics['val_log_likelihood']) + 1)
        axes[2].plot(val_epochs, metrics['val_log_likelihood'], label='Val Log-Likelihood', color='orange')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Log-Likelihood')
        axes[2].set_title('Validation Log-Likelihood')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_reconstructions(self, 
                                model: VariationalAutoencoder,
                                data_loader: DataLoader,
                                num_samples: int = 8) -> None:
        """Visualize original images and their reconstructions."""
        model.eval()
        
        with torch.no_grad():
            images, _ = next(iter(data_loader))
            images = images[:num_samples]
            reconstructions = model.reconstruct(images)
            
            images_np = images.cpu().numpy()
            reconstructions_np = reconstructions.cpu().view(-1, self.image_size, self.image_size).numpy()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Original images
            axes[0, i].imshow(images_np[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructions
            axes[1, i].imshow(reconstructions_np[i], cmap='gray')
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_generated_samples(self, 
                                  model: VariationalAutoencoder,
                                  num_samples: int = 8,
                                  device: str = 'cpu') -> None:
        """Visualize samples generated from latent space."""
        generated_samples = model.sample(num_samples, device)
        generated_images = generated_samples.cpu().view(-1, self.image_size, self.image_size).numpy()
        
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            axes[i].imshow(generated_images[i], cmap='gray')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_latent_space_interpolation(self, 
                                      model: VariationalAutoencoder,
                                      start_image: torch.Tensor,
                                      end_image: torch.Tensor,
                                      num_steps: int = 10,
                                      device: str = 'cpu') -> None:
        """Visualize interpolation in latent space between two images."""
        model.eval()
        
        with torch.no_grad():
            # Encode start and end images
            start_flat = start_image.view(-1, model.encoder.fc1.in_features).to(device)
            end_flat = end_image.view(-1, model.encoder.fc1.in_features).to(device)
            
            start_mean, start_logvar = model.encoder(start_flat)
            end_mean, end_logvar = model.encoder(end_flat)
            
            # Interpolate in latent space
            alphas = np.linspace(0, 1, num_steps)
            interpolated_images = []
            
            for alpha in alphas:
                interpolated_mean = (1 - alpha) * start_mean + alpha * end_mean
                interpolated_logvar = (1 - alpha) * start_logvar + alpha * end_logvar
                
                z = model.reparameterize(interpolated_mean, interpolated_logvar)
                interpolated_image = model.decoder(z)
                interpolated_images.append(interpolated_image.cpu().view(self.image_size, self.image_size))
        
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img.numpy(), cmap='gray')
            axes[i].set_title(f'α={alphas[i]:.1f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate the VAE framework."""
    print("=== Variational Autoencoder Framework ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare data
    print("\n1. Preparing MNIST data...")
    data_manager = MNISTDataManager(image_size=14)
    train_loader, val_loader = data_manager.get_datasets(
        num_images_per_class=1000,
        train_split=0.9
    )
    
    # 2. Create and train VAE
    print("\n2. Creating and training VAE...")
    vae = VariationalAutoencoder(
        input_dim=14*14,  # 14x14 images
        hidden_dim=128,
        latent_dim=8
    )
    
    trainer = VAETrainer(device)
    metrics = trainer.train_model(
        vae, train_loader, val_loader,
        epochs=20,  # Reduced for demo
        lr=0.001
    )
    
    # 3. Visualize training
    print("\n3. Visualizing training progress...")
    visualizer = VAEVisualizer(image_size=14)
    visualizer.plot_training_curves(metrics)
    
    # 4. Visualize reconstructions
    print("\n4. Visualizing reconstructions...")
    visualizer.visualize_reconstructions(vae, val_loader, num_samples=8)
    
    # 5. Generate new samples
    print("\n5. Generating new samples...")
    visualizer.visualize_generated_samples(vae, num_samples=8, device=device)
    
    # 6. Latent space interpolation
    print("\n6. Latent space interpolation...")
    sample_images, _ = next(iter(val_loader))
    start_image = sample_images[0:1]
    end_image = sample_images[1:2]
    
    visualizer.plot_latent_space_interpolation(
        vae, start_image, end_image, num_steps=8, device=device
    )
    
    # 7. Final metrics
    print("\n7. Final Results:")
    print(f"Final training loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final reconstruction loss: {metrics['reconstruction_loss'][-1]:.4f}")
    print(f"Final KL divergence: {metrics['kl_divergence'][-1]:.4f}")
    print(f"Final validation log-likelihood: {metrics['val_log_likelihood'][-1]:.4f}")
    
    print("\nVAE framework demonstration complete!")


if __name__ == "__main__":
    main()