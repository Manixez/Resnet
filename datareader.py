import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import (
    ToTensor, Normalize, Compose, RandomResizedCrop, 
    RandomHorizontalFlip, ColorJitter, Resize
)
from sklearn.model_selection import StratifiedShuffleSplit

# Set random seed for reproducibility
RANDOM_SEED = 2025
random.seed(RANDOM_SEED)      # random seed Python
np.random.seed(RANDOM_SEED)   # random seed Numpy
torch.manual_seed(RANDOM_SEED) # random seed PyTorch
torch.cuda.manual_seed(RANDOM_SEED) if torch.cuda.is_available() else None  # GPU seed



class MakananIndo(Dataset):
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 data_dir='train',
                 img_size=(224, 224),  # Update default size for ViT
                 transform=None,
                 split='train'
                 ):
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
        # Default transforms with data augmentation for training
        if transform is None:
            if split == 'train':
                self.transform = Compose([
                    RandomResizedCrop(img_size, scale=(0.9, 1.0)),  # Less aggressive crop
                    RandomHorizontalFlip(p=0.5),  # Horizontal flip
                    ColorJitter(
                        brightness=0.1,  # Reduced brightness variation
                        contrast=0.1,    # Reduced contrast variation
                        saturation=0.1   # Reduced saturation variation
                    ),
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
            else:  # validation/test transforms
                self.transform = Compose([
                    Resize(img_size),  # Simple resize to target size
                    ToTensor(),
                    Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
                ])
        else:
            self.transform = transform

        # Baca CSV terlebih dahulu
        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        
        # Verifikasi bahwa semua file ada
        all_files = set(os.listdir(data_dir))
        missing_files = [f for f in df['filename'] if f not in all_files]
        if missing_files:
            raise ValueError(f'Files missing in {data_dir}: {missing_files}')
        
        # Gunakan hanya file yang ada di CSV
        self.image_files = df['filename'].tolist()
        self.labels = df['label'].tolist()
        
        # Verifikasi integritas data
        for img_file, label in zip(self.image_files, self.labels):
            img_path = os.path.join(data_dir, img_file)
            if not os.path.exists(img_path):
                raise ValueError(f'Image not found: {img_path}')
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        print(f'Warning: Converting {img_file} from {img.mode} to RGB')
            except Exception as e:
                raise ValueError(f'Error reading {img_path}: {str(e)}')
        
        # Pasangkan setiap image file dengan labelnya dan simpan dalam list of tuples
        all_data = list(zip(self.image_files, self.labels))
        
        # Menggunakan stratified split untuk memastikan distribusi kelas seimbang
        X = list(range(len(all_data)))
        y = [label for _, label in all_data]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_indices, val_indices = next(sss.split(X, y))
        
        if split == 'train':
            self.data = [all_data[i] for i in train_indices]
        elif split == 'val':
            self.data = [all_data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'")
        
        # Define default transforms
        self.default_transform = Compose([
            ToTensor(),  # Converts PIL/numpy to tensor and scales to [0,1]
            Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # Load Data dan Label
            img_name = self.data[idx][0]
            img_path = os.path.join(self.data_dir, img_name)
            
            # Load image using PIL directly (more reliable than cv2 for this case)
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms (including resize and normalization)
            try:
                if self.transform:
                    image = self.transform(image)
                else:
                    image = self.default_transform(image)
            except Exception as e:
                raise ValueError(f'Error applying transforms to {img_name}: {str(e)}')
            
            label = self.data[idx][1]
            
            # return data gambar, label, dan nama file (bukan path penuh)
            return image, label, img_name
            
        except Exception as e:
            print(f'Error processing image at index {idx}: {str(e)}')
            # Return default/placeholder data in case of error
            empty_tensor = torch.zeros((3, *self.img_size))
            return empty_tensor, -1, 'error'

if __name__ == "__main__":
    # Test both splits
    train_dataset = MakananIndo(split='train')
    val_dataset = MakananIndo(split='val')
    
    print(f"Train data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    
    # Sample 5 random images from each dataset
    train_indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    
    print("\nTrain Dataset Samples:")
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        print(f"Train data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)
    
    print("\nValidation Dataset Samples:")
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        print(f"Val data ke-{i} (index {idx})")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"File path: {filepath}")
        print("-" * 40)

    # Create a figure with subplots for both train and val
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot train images
    for i, idx in enumerate(train_indices):
        image, label, filepath = train_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * train_dataset.IMAGENET_STD[j] + train_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Train: {label}")
        axes[0, i].axis('off')
    
    # Plot val images
    for i, idx in enumerate(val_indices):
        image, label, filepath = val_dataset[idx]
        
        # Convert tensor to displayable format
        img_display = image.clone()
        for j in range(3):
            img_display[j] = img_display[j] * val_dataset.IMAGENET_STD[j] + val_dataset.IMAGENET_MEAN[j]
        
        img_display = img_display.permute(1, 2, 0)
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[1, i].imshow(img_display)
        axes[1, i].set_title(f"Val: {label}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('cek_augmentasi.png') # <-- Ganti dengan baris ini
    plt.close()