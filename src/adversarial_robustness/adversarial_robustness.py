#!/usr/bin/env python3
"""
Adversarial Robustness Framework

A comprehensive framework for training CNN models with adversarial robustness evaluation,
including FGSM attacks, data augmentation, and gradient visualization.

Author: Extracted from ESE5460 HW2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Callable


class AllCNN(nn.Module):
    """All-CNN architecture for CIFAR-10 classification."""
    
    def __init__(self, c1: int = 96, c2: int = 192):
        super().__init__()
        d = 0.5  # Dropout rate
        
        def convbn(ci: int, co: int, ksz: int, s: int = 1, pz: int = 0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co)
            )
        
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3, c1, 3, 1, 1),
            convbn(c1, c1, 3, 1, 1),
            convbn(c1, c1, 3, 2, 1),
            nn.Dropout(d),
            convbn(c1, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 2, 1),
            nn.Dropout(d),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, 10, 1, 1),
            nn.AvgPool2d(8),
            View(10)
        )
        
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class View(nn.Module):
    """Reshape layer for flattening."""
    
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.output_size)


class DataManager:
    """Data loading and augmentation manager."""
    
    def __init__(self, data_dir: str = './data', batch_size: int = 16):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
    
    def get_basic_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get basic CIFAR-10 data loaders without augmentation."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False
        )
        
        return trainloader, testloader
    
    def get_augmented_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get CIFAR-10 data loaders with data augmentation."""
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False
        )
        
        return trainloader, testloader


class Trainer:
    """Model training manager with learning rate scheduling."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def lr_lambda(self, epoch: int) -> float:
        """Learning rate schedule."""
        if epoch < 40:
            return 1.0
        elif 40 <= epoch < 80:
            return 0.1
        else:
            return 0.01
    
    def train_model(self, 
                   model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   epochs: int = 100,
                   lr: float = 0.01,
                   momentum: float = 0.9,
                   weight_decay: float = 0.001,
                   model_name: str = 'model',
                   use_tensorboard: bool = True) -> Dict[str, List[float]]:
        """
        Train a model with the specified parameters.
        
        Returns:
            Dictionary containing training metrics
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                            weight_decay=weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        
        if use_tensorboard:
            logger = SummaryWriter(log_dir=f'runs/{model_name}')
        
        train_loss_values = []
        train_error = []
        val_loss_values = []
        val_error = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if use_tensorboard and (i + 1) % 1000 == 0:
                    logger.add_scalar(f'loss_{model_name}', loss.item(), epoch * len(train_loader) + i)
            
            avg_train_loss = running_loss / total
            train_loss_values.append(avg_train_loss)
            train_error.append(100 - 100 * correct / total)
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_running_loss / total
            val_loss_values.append(avg_val_loss)
            val_error.append(100 - 100 * correct / total)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {100 * (total - (train_error[-1] * total / 100)) / total:.2f}%')
                print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {100 * correct / total:.2f}%')
                print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        if use_tensorboard:
            logger.close()
        
        return {
            'train_loss': train_loss_values,
            'train_error': train_error,
            'val_loss': val_loss_values,
            'val_error': val_error
        }


class AdversarialAttacker:
    """FGSM adversarial attack implementation."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def fgsm_attack(self, 
                   data_loader: torch.utils.data.DataLoader,
                   eps: float = 8/255,
                   num_steps: int = 5) -> List[float]:
        """
        Perform multi-step FGSM attack.
        
        Args:
            data_loader: Data loader for test images
            eps: Perturbation magnitude per step
            num_steps: Number of attack steps
            
        Returns:
            List of loss values at each step
        """
        self.model.eval()
        loss_values_per_step = np.zeros(num_steps)
        
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad_()
            
            perturbed_images = images.clone().detach()
            step_losses = np.zeros(num_steps)
            
            for k in range(num_steps):
                perturbed_images.requires_grad_()
                outputs = self.model(perturbed_images)
                loss = self.criterion(outputs, labels)
                
                if perturbed_images.grad is not None:
                    perturbed_images.grad.zero_()
                
                loss.backward()
                dx = perturbed_images.grad.sign()
                perturbed_images = perturbed_images + eps * dx
                perturbed_images = torch.clamp(perturbed_images, 0, 1)
                perturbed_images = perturbed_images.detach()
                
                outputs = self.model(perturbed_images)
                perturbed_loss = self.criterion(outputs, labels)
                step_losses[k] += perturbed_loss.item()
            
            loss_values_per_step += step_losses
            break  # Use only one batch for demonstration
        
        return loss_values_per_step.tolist()
    
    def evaluate_robustness(self, 
                          data_loader: torch.utils.data.DataLoader,
                          eps: float = 8/255) -> Tuple[float, float]:
        """
        Evaluate model robustness against 1-step FGSM attack.
        
        Returns:
            Tuple of (clean_accuracy, adversarial_accuracy)
        """
        self.model.eval()
        
        total_clean = 0
        correct_clean = 0
        total_adv = 0
        correct_adv = 0
        
        # Clean accuracy
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        
        # Adversarial accuracy
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad_()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            
            dx = images.grad.sign()
            perturbed_images = images + eps * dx
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            
            outputs_adv = self.model(perturbed_images)
            _, predicted_adv = torch.max(outputs_adv.data, 1)
            total_adv += labels.size(0)
            correct_adv += (predicted_adv == labels).sum().item()
        
        clean_acc = 100 * correct_clean / total_clean
        adv_acc = 100 * correct_adv / total_adv
        
        return clean_acc, adv_acc


class GradientVisualizer:
    """Visualization utilities for gradients and adversarial examples."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def visualize_gradients(self, 
                          data_loader: torch.utils.data.DataLoader,
                          output_dir: str = 'output_plots',
                          num_correct: int = 5,
                          num_incorrect: int = 5) -> None:
        """
        Visualize gradients for correctly and incorrectly classified images.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.eval()
        correct_count = 0
        incorrect_count = 0
        
        for images, labels in data_loader:
            if correct_count >= num_correct and incorrect_count >= num_incorrect:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad_()
            
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            dx = images.grad.clone()
            
            for j in range(images.size(0)):
                if predicted[j] == labels[j] and correct_count < num_correct:
                    self._save_image_and_gradient(
                        images[j], dx[j], output_dir, 
                        f'correct_classified_image_{correct_count}.png',
                        'Correctly Classified Image'
                    )
                    correct_count += 1
                elif predicted[j] != labels[j] and incorrect_count < num_incorrect:
                    self._save_image_and_gradient(
                        images[j], dx[j], output_dir,
                        f'misclassified_image_{incorrect_count}.png',
                        'Misclassified Image'
                    )
                    incorrect_count += 1
    
    def _save_image_and_gradient(self, 
                               image: torch.Tensor,
                               gradient: torch.Tensor,
                               output_dir: str,
                               filename: str,
                               title: str) -> None:
        """Save image and its gradient visualization."""
        original_image = image.detach().cpu().numpy()
        grad_image = gradient.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(np.transpose(original_image, (1, 2, 0)))
        axes[0].set_title(title)
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        
        # Gradient
        axes[1].imshow(grad_image[0], cmap='hot')
        axes[1].set_title('Gradient of Loss w.r.t. Input')
        axes[1].set_aspect('equal')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def plot_training_curves(metrics: Dict[str, List[float]], 
                        title: str = "Training Curves",
                        save_path: Optional[str] = None) -> None:
    """Plot training and validation curves."""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training and validation loss
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Training Loss')
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation error
    axes[0, 1].plot(epochs, metrics['train_error'], label='Training Error')
    axes[0, 1].plot(epochs, metrics['val_error'], label='Validation Error')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Error (%)')
    axes[0, 1].set_title('Training and Validation Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss (log scale)
    axes[1, 0].semilogy(epochs, metrics['train_loss'], label='Training Loss')
    axes[1, 0].semilogy(epochs, metrics['val_loss'], label='Validation Loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Training and Validation Loss (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final accuracies
    final_train_acc = 100 - metrics['train_error'][-1]
    final_val_acc = 100 - metrics['val_error'][-1]
    axes[1, 1].bar(['Training', 'Validation'], [final_train_acc, final_val_acc])
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Final Accuracies')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Demonstrate the adversarial robustness framework."""
    print("=== Adversarial Robustness Framework ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare data
    print("\n1. Preparing data...")
    data_manager = DataManager(batch_size=16)
    train_loader, test_loader = data_manager.get_basic_loaders()
    aug_train_loader, aug_test_loader = data_manager.get_augmented_loaders()
    
    # 2. Train baseline model
    print("\n2. Training baseline model...")
    model = AllCNN()
    trainer = Trainer(device)
    
    metrics = trainer.train_model(
        model, train_loader, test_loader,
        epochs=20,  # Reduced for demo
        model_name='baseline'
    )
    
    plot_training_curves(metrics, "Baseline Model Training")
    
    # 3. Train with data augmentation
    print("\n3. Training with data augmentation...")
    aug_model = AllCNN()
    aug_metrics = trainer.train_model(
        aug_model, aug_train_loader, aug_test_loader,
        epochs=20,  # Reduced for demo
        model_name='augmented'
    )
    
    plot_training_curves(aug_metrics, "Augmented Model Training")
    
    # 4. Adversarial evaluation
    print("\n4. Evaluating adversarial robustness...")
    attacker = AdversarialAttacker(aug_model, device)
    
    clean_acc, adv_acc = attacker.evaluate_robustness(aug_test_loader)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    
    # 5. Multi-step attack analysis
    print("\n5. Multi-step attack analysis...")
    attack_losses = attacker.fgsm_attack(aug_test_loader, num_steps=5)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(attack_losses) + 1), attack_losses, 'o-')
    plt.xlabel('Attack Step')
    plt.ylabel('Loss')
    plt.title('Loss on Perturbed Images (Multi-step FGSM)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 6. Gradient visualization
    print("\n6. Visualizing gradients...")
    visualizer = GradientVisualizer(aug_model, device)
    visualizer.visualize_gradients(aug_test_loader)
    
    print("\nFramework demonstration complete!")


if __name__ == "__main__":
    main()