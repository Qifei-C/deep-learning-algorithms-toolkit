"""
MNIST Sample Data Generator
Generates synthetic MNIST-like data for testing
"""

import numpy as np
import os

def generate_mnist_sample(num_samples=1000, image_size=28):
    """
    Generate synthetic MNIST-like data
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of square images
        
    Returns:
        Tuple of (images, labels)
    """
    # Generate random images with simple patterns
    images = np.zeros((num_samples, image_size, image_size))
    labels = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        label = i % 10  # Cycle through digits 0-9
        labels[i] = label
        
        # Create simple patterns for each digit
        img = np.zeros((image_size, image_size))
        center_x, center_y = image_size // 2, image_size // 2
        
        if label == 0:  # Circle
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= (image_size//3)**2
            img[mask] = 0.8
            inner_mask = (x - center_x)**2 + (y - center_y)**2 <= (image_size//5)**2
            img[inner_mask] = 0.0
            
        elif label == 1:  # Vertical line
            img[:, center_x-2:center_x+3] = 0.8
            
        elif label == 2:  # Horizontal lines
            img[center_y-6:center_y-3, 5:-5] = 0.8
            img[center_y:center_y+3, 5:-5] = 0.8
            img[center_y+6:center_y+9, 5:-5] = 0.8
            
        elif label == 3:  # Right-side curves
            img[center_y-8:center_y-5, 5:15] = 0.8
            img[center_y-2:center_y+2, 8:18] = 0.8
            img[center_y+5:center_y+8, 5:15] = 0.8
            
        elif label == 4:  # L-shape
            img[5:15, center_x-2:center_x+1] = 0.8
            img[center_y-1:center_y+2, 5:20] = 0.8
            img[15:25, 15:18] = 0.8
            
        elif label == 5:  # S-shape
            img[5:8, 5:20] = 0.8
            img[8:15, 5:8] = 0.8
            img[12:15, 8:18] = 0.8
            img[15:22, 15:18] = 0.8
            img[19:22, 5:18] = 0.8
            
        elif label == 6:  # P-shape
            img[5:22, 5:8] = 0.8
            img[5:8, 8:18] = 0.8
            img[12:15, 8:15] = 0.8
            
        elif label == 7:  # Triangle
            for row in range(5, 20):
                width = (row - 5) // 2
                start = center_x - width
                end = center_x + width + 1
                img[row, start:end] = 0.8
                
        elif label == 8:  # Double circle
            y, x = np.ogrid[:image_size, :image_size]
            # Upper circle
            mask1 = (x - center_x)**2 + (y - center_y + 5)**2 <= 40
            img[mask1] = 0.8
            inner1 = (x - center_x)**2 + (y - center_y + 5)**2 <= 20
            img[inner1] = 0.0
            # Lower circle
            mask2 = (x - center_x)**2 + (y - center_y - 5)**2 <= 40
            img[mask2] = 0.8
            inner2 = (x - center_x)**2 + (y - center_y - 5)**2 <= 20
            img[inner2] = 0.0
            
        elif label == 9:  # Spiral
            angles = np.linspace(0, 4*np.pi, 100)
            for t in angles:
                r = t * 2
                x = int(center_x + r * np.cos(t))
                y = int(center_y + r * np.sin(t))
                if 0 <= x < image_size and 0 <= y < image_size:
                    img[max(0, y-1):min(image_size, y+2), 
                        max(0, x-1):min(image_size, x+2)] = 0.8
        
        # Add noise
        noise = np.random.normal(0, 0.1, (image_size, image_size))
        img = np.clip(img + noise, 0, 1)
        
        images[i] = img
    
    return images, labels

def save_mnist_sample(filepath, num_samples=1000):
    """
    Generate and save MNIST sample data
    
    Args:
        filepath: Path to save the data
        num_samples: Number of samples to generate
    """
    images, labels = generate_mnist_sample(num_samples)
    
    # Save as numpy arrays
    np.savez(filepath, images=images, labels=labels)
    print(f"Saved {num_samples} MNIST-like samples to {filepath}")

if __name__ == "__main__":
    # Generate sample data
    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, "mnist_sample.npz")
    save_mnist_sample(save_path, 1000)