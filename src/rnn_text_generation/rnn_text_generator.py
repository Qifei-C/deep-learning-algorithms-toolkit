#!/usr/bin/env python3
"""
RNN Text Generation Framework

A comprehensive framework for character-level text generation using LSTM networks,
including data preprocessing, model training, and text generation capabilities.

Author: Extracted from ESE5460 HW3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import re
from collections import Counter


class TextProcessor:
    """Text preprocessing and data management utilities."""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = data_dir
        self.vocab = None
        self.vocab_size = 0
        self.char_to_index = {}
        self.index_to_char = {}
        
    def download_texts(self) -> None:
        """Download classic texts from Project Gutenberg."""
        urls = {
            "shakespeare.txt": "https://www.gutenberg.org/ebooks/100.txt.utf-8",
            "war_and_peace.txt": "https://www.gutenberg.org/ebooks/2600.txt.utf-8"
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        for filename, url in urls.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(filepath, 'wb') as file:
                        file.write(response.content)
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
    
    def load_and_combine_texts(self) -> str:
        """Load all text files and combine into single corpus."""
        combined_text = ""
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                        combined_text += text + " "
                    print(f"Loaded: {filename} ({len(text)} characters)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return combined_text
    
    def build_vocabulary(self, text: str) -> None:
        """Build character vocabulary from text."""
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {self.vocab[:20]}")
    
    def text_to_sequences(self, 
                         text: str, 
                         sequence_length: int = 32,
                         stride: int = None) -> List[List[int]]:
        """Convert text to sequences of character indices."""
        if stride is None:
            stride = sequence_length
            
        sequences = []
        for i in range(0, len(text) - sequence_length + 1, stride):
            sequence = text[i:i + sequence_length]
            sequence_indices = [self.char_to_index.get(char, 0) for char in sequence]
            sequences.append(sequence_indices)
        
        return sequences
    
    def one_hot_encode_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """Convert sequences to one-hot encoded tensors."""
        encoded_sequences = []
        
        for sequence in sequences:
            one_hot_sequence = []
            for char_idx in sequence:
                one_hot_vector = torch.zeros(self.vocab_size)
                one_hot_vector[char_idx] = 1.0
                one_hot_sequence.append(one_hot_vector)
            encoded_sequences.append(torch.stack(one_hot_sequence))
        
        return torch.stack(encoded_sequences)
    
    def create_datasets(self, 
                       text: str,
                       sequence_length: int = 32,
                       train_split: float = 0.7,
                       val_split: float = 0.15) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        """Create train, validation, and test datasets."""
        sequences = self.text_to_sequences(text, sequence_length)
        
        # Create input-target pairs for language modeling
        input_sequences = []
        target_sequences = []
        
        for seq in sequences:
            if len(seq) > 1:
                input_sequences.append(seq[:-1])  # All but last character
                target_sequences.append(seq[1:])   # All but first character
        
        # Convert to tensors
        input_tensor = self.one_hot_encode_sequences(input_sequences)
        target_tensor = torch.tensor(target_sequences, dtype=torch.long)
        
        # Split data
        total_size = len(input_tensor)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = TensorDataset(input_tensor[train_indices], target_tensor[train_indices])
        val_dataset = TensorDataset(input_tensor[val_indices], target_tensor[val_indices])
        test_dataset = TensorDataset(input_tensor[test_indices], target_tensor[test_indices])
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset


class CharacterLSTM(nn.Module):
    """LSTM model for character-level language modeling."""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        print(f"Model parameters: {self.count_parameters():,}")
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass through the LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, vocab_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            output: Predictions of shape (batch_size, sequence_length, vocab_size)
            hidden: Updated hidden state tuple
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h_0, c_0)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.output_layer(lstm_out)
        
        return output, hidden
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextGenerator:
    """Text generation utilities."""
    
    def __init__(self, 
                 model: CharacterLSTM,
                 processor: TextProcessor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device)
    
    def generate_text(self, 
                     seed_text: str,
                     length: int = 100,
                     temperature: float = 1.0) -> str:
        """
        Generate text using trained model.
        
        Args:
            seed_text: Starting text
            length: Number of characters to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        self.model.eval()
        
        # Prepare seed
        current_text = seed_text
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                # Encode current character
                if current_text:
                    last_char = current_text[-1]
                    if last_char in self.processor.char_to_index:
                        char_idx = self.processor.char_to_index[last_char]
                        input_tensor = torch.zeros(1, 1, self.processor.vocab_size, device=self.device)
                        input_tensor[0, 0, char_idx] = 1.0
                    else:
                        # Use random character if not in vocabulary
                        char_idx = np.random.randint(0, self.processor.vocab_size)
                        input_tensor = torch.zeros(1, 1, self.processor.vocab_size, device=self.device)
                        input_tensor[0, 0, char_idx] = 1.0
                else:
                    # Start with random character
                    char_idx = np.random.randint(0, self.processor.vocab_size)
                    input_tensor = torch.zeros(1, 1, self.processor.vocab_size, device=self.device)
                    input_tensor[0, 0, char_idx] = 1.0
                
                # Forward pass
                output, hidden = self.model(input_tensor, hidden)
                
                # Apply temperature and sample
                logits = output[0, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probabilities, 1).item()
                
                # Convert to character and append
                next_char = self.processor.index_to_char[next_char_idx]
                current_text += next_char
        
        return current_text
    
    def sample_with_diversity(self, 
                            seed_text: str,
                            length: int = 100,
                            temperatures: List[float] = [0.5, 1.0, 1.5]) -> Dict[float, str]:
        """Generate text with different temperature settings."""
        results = {}
        for temp in temperatures:
            results[temp] = self.generate_text(seed_text, length, temp)
        return results


class RNNTrainer:
    """Training manager for RNN models."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def train_model(self,
                   model: CharacterLSTM,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 50,
                   lr: float = 0.001,
                   clip_grad: float = 5.0) -> Dict[str, List[float]]:
        """
        Train the RNN model.
        
        Returns:
            Dictionary containing training metrics
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        train_perplexities = []
        val_perplexities = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs, _ = model(batch_inputs)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, model.vocab_size)
                targets = batch_targets.reshape(-1)
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            train_perplexities.append(np.exp(avg_train_loss))
            
            # Validation phase
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs, _ = model(batch_inputs)
                    outputs = outputs.reshape(-1, model.vocab_size)
                    targets = batch_targets.reshape(-1)
                    
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            val_perplexities.append(np.exp(avg_val_loss))
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}, Perplexity: {train_perplexities[-1]:.2f}')
                print(f'  Val Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexities[-1]:.2f}')
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_perplexity': train_perplexities,
            'val_perplexity': val_perplexities
        }


def plot_training_metrics(metrics: Dict[str, List[float]], 
                         title: str = "Training Metrics") -> None:
    """Plot training and validation metrics."""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(epochs, metrics['train_loss'], label='Training Loss', color='blue')
    axes[0].plot(epochs, metrics['val_loss'], label='Validation Loss', color='red')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity curves
    axes[1].plot(epochs, metrics['train_perplexity'], label='Training Perplexity', color='green')
    axes[1].plot(epochs, metrics['val_perplexity'], label='Validation Perplexity', color='orange')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    """Demonstrate the RNN text generation framework."""
    print("=== RNN Text Generation Framework ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare data
    print("\n1. Preparing text data...")
    processor = TextProcessor()
    processor.download_texts()
    
    combined_text = processor.load_and_combine_texts()
    processor.build_vocabulary(combined_text)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = processor.create_datasets(
        combined_text, sequence_length=32
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    
    # 2. Create and train model
    print("\n2. Creating and training model...")
    model = CharacterLSTM(
        vocab_size=processor.vocab_size,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    trainer = RNNTrainer(device)
    metrics = trainer.train_model(
        model, train_loader, val_loader,
        epochs=20,  # Reduced for demo
        lr=0.001
    )
    
    plot_training_metrics(metrics, "Character-Level LSTM Training")
    
    # 3. Generate text
    print("\n3. Generating text...")
    generator = TextGenerator(model, processor, device)
    
    seed_texts = ["The quick brown", "Once upon a time", "In the beginning"]
    
    for seed in seed_texts:
        print(f"\nSeed: '{seed}'")
        
        # Generate with different temperatures
        diverse_samples = generator.sample_with_diversity(
            seed, length=200, temperatures=[0.5, 1.0, 1.5]
        )
        
        for temp, text in diverse_samples.items():
            print(f"\nTemperature {temp}:")
            print(f"'{text}'")
            print("-" * 80)
    
    # 4. Model evaluation
    print("\n4. Model evaluation...")
    final_train_perplexity = metrics['train_perplexity'][-1]
    final_val_perplexity = metrics['val_perplexity'][-1]
    
    print(f"Final Training Perplexity: {final_train_perplexity:.2f}")
    print(f"Final Validation Perplexity: {final_val_perplexity:.2f}")
    
    print("\nFramework demonstration complete!")


if __name__ == "__main__":
    main()