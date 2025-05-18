"""
INPUT DATA
1) a list of s2_blocks, which are a 3d tensor of concatenated Sentinel 2 time series, including the RGB, NIR, and NDVI bands for a 81x81 block of pixels 

2) a list of corresponding land cover classes from the USFS dataset at the target pixel's location

OUTPUT
1) An integer representing the class that the model thinks applies to the target pixel
"""
import os

import sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

model_dir = os.path.dirname(os.path.abspath('__file__'))
src_dir = Path(model_dir).parent
root_dir = Path(src_dir).parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src.data.CNN.cnn_data_generator import generate_time_series_training_samples, generate_roi_list

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# USFS Land Cover Class Mapping 
class_names = {
    1: "Trees",
    2: "Tall Shrubs & Trees Mix (SEAK Only)", 
    3: "Shrubs & Trees Mix",
    4: "Grass/Forb/Herb & Trees Mix",
    5: "Barren & Trees Mix",
    6: "Tall Shrubs (SEAK Only)",
    7: "Shrubs",
    8: "Grass/Forb/Herb & Shrubs Mix",
    9: "Barren & Shrubs Mix",
    10: "Grass/Forb/Herb",
    11: "Barren & Grass/Forb/Herb Mix",
    12: "Barren or Impervious",
    13: "Snow or Ice",
    14: "Water",
    15: "Non-Processing Area Mask"
}


# Custom Dataset Class
class LandCoverDataset(Dataset):
    # Land cover dataset is the same as the previous CNN
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] - 1
        image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
            
        return image, label


class DiskLandCoverDataset(Dataset):
    """Dataset for loading satellite time series from disk."""
    
    def __init__(self, metadata_path, normalize=True, transform=None):
        """
        Initialize dataset from metadata file.
        
        Parameters:
        -----------
        metadata_path : str
            Path to metadata JSON file
        normalize : bool
            Whether to normalize blocks (divide by 10000)
        transform : callable
            Optional transform to apply to the data
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.block_paths = metadata['blocks']
        self.classes = metadata['classes']
        self.roi_names = metadata['roi_names']
        self.target_pixels = metadata['target_pixels']
        self.normalize = normalize
        self.transform = transform
        
        # Check number of time points (first block)
        first_block_dir = self.block_paths[0]
        self.n_timepoints = len([f for f in os.listdir(first_block_dir) if f.startswith('time_')])
        
        print(f"Initialized dataset with {len(self.block_paths)} samples across {self.n_timepoints} time points")
    
    def __len__(self):
        return len(self.block_paths)
    
    def __getitem__(self, idx):
        """Load a block from disk and process it."""
        # Get block directory
        block_dir = self.block_paths[idx]
        
        # Load time point blocks and concatenate
        time_path = os.path.join(block_dir, f"time_0.npy")
        block = np.load(time_path)
        block = block[:6]
           
        # Normalize if requested
        if self.normalize:
            block = block.astype(np.float32) / 10000.0
        
        # Convert label (subtract 1 for 0-indexed classes)
        label = self.classes[idx] - 1
        
        # Convert to tensor
        block_tensor = torch.tensor(block, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            block_tensor = self.transform(block_tensor)
        
        return block_tensor, label


class SpatialAttention(nn.Module):
    """
    Spatial attention module that emphasizes the center of the input feature maps.
    Combines a learned attention with a center-biased prior.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Conv layer to generate attention map
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate center-biased prior (Gaussian-like)
        batch, channels, height, width = x.size()
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing='ij'
        )
        
        distance_squared = x_grid**2 + y_grid**2
        center_prior = torch.exp(-distance_squared / 0.5)
        center_prior = center_prior.unsqueeze(0).unsqueeze(0)

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        learned_attention = self.sigmoid(self.conv(pooled))
        combined_attention = learned_attention * center_prior
        
        return combined_attention


class LandCoverCNN(nn.Module):
    def __init__(self, num_classes, input_channels=None, block_size=80):
        super(LandCoverCNN, self).__init__()
        
        # Auto-detect input channels if not specified
        if input_channels is None:
            # Base channels per time point (RGB+NIR+SWIR1+SWIR2+indices)
            input_channels = 6  # 6 raw bands + 5 indices
        
        # Input: input_channels × block_size × block_size
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.spatial_attention2 = SpatialAttention(kernel_size=5)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.spatial_attention1 = SpatialAttention(kernel_size=3)
        
        # Calculate feature dimensions after pooling
        feature_size = block_size // 8
        if block_size % 8 != 0:
            feature_size = feature_size + 1
        
        self.feature_size = feature_size
        flattened_size = 256 * feature_size * feature_size
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        print(f"Model initialized with {input_channels} input channels")
        print(f"Feature map after pooling: 256 × {feature_size} × {feature_size}")
        print(f"Flattened features: {flattened_size}")
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Main training function
def train_land_cover_model(dataset_info, batch_size=32, epochs=50, learning_rate=0.001, block_size=80):
    """
    Train CNN model with disk-based dataset.
    
    Parameters:
    -----------
    dataset_info : dict
        Information about the dataset from generate_time_series_training_samples
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    block_size : int
        Size of input blocks
        
    Returns:
    --------
    model : LandCoverCNN
        Trained model
    history : dict
        Training history
    """
    # Load metadata
    metadata_path = dataset_info['metadata_path']
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get class distribution for stratification
    classes = np.array(metadata['classes'])
    
    # Create train/val split indices
    train_indices, val_indices = train_test_split(
        np.arange(len(classes)), 
        test_size=0.2, 
        stratify=classes, 
        random_state=42
    )
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    # Create datasets with indices
    full_dataset_train = DiskLandCoverDataset(metadata_path, normalize=True, transform=train_transform)
    full_dataset_val = DiskLandCoverDataset(metadata_path, normalize=True, transform=None)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Get first sample to determine input channels
    sample_data, _ = full_dataset_val[0]  # Use val dataset without transforms
    n_channels = sample_data.shape[0]
    print(f"Input has {n_channels} channels")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    total_possible_classes = 15
    model = LandCoverCNN(total_possible_classes, input_channels=n_channels, block_size=block_size).to(device)

    # Get training labels by extracting the classes for the training indices
    y_train = classes[train_indices]
    class_counts = np.bincount(y_train, minlength=total_possible_classes+1)[1:]  # Skip index 0 since classes start at 1
    
    # Ensure there are no zeros in class_counts (which would cause division by zero)
    min_count = 1  # Minimum count to avoid division by zero
    class_counts = np.maximum(class_counts, min_count)
    
    # Create class weights
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 4. Training Loop
    print("Starting training...")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    torch.cuda.empty_cache()
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Reset gradients at epoch start
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels) / 2  # If using gradient accumulation
            
            # Use scaler for mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Update every 2 batches (gradient accumulation)
            if (i + 1) % 2 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track statistics
            train_loss += loss.item() * 2  # Adjust for accumulation
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
                
            # Clear cache occasionally
            if i % 50 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'block_size': block_size,
                'num_classes': 11,
                # Any other parameters you might need
            }, 'time_series_80x80_CNN_block_model.pth')   # TODO: UPDATE PATH NAME
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # 5. Load the best model for evaluation
    # Load the checkpoint
    checkpoint = torch.load('time_series_80x80_CNN_block_model.pth')  # TODO: UPDATE PATH NAME

    # Initialize model with the saved parameters
    model = LandCoverCNN(
        num_classes=11,  # TODO: IDK THIS WORKS WHEN I SET IT TO 11, INSTEAD OF READING THE SAVED NUMBER OF UNIQUE CLASSES READ DURING MODEL TRAINING???
        block_size=checkpoint['block_size']
    )

    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


# Evaluation functions
def evaluate_model(model, s2_blocks, classes):
    """
    Evaluate the trained model on test data
    
    Parameters:
    -----------
    model : LandCoverCNN
        Trained model
    s2_blocks : numpy.ndarray
        Test image blocks
    classes : numpy.ndarray
        True class labels
        
    Returns:
    --------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    """
    # Normalize the data
    s2_blocks = s2_blocks.astype(np.float32) / 10000.0
    
    # Create dataset and dataloader
    test_dataset = LandCoverDataset(s2_blocks, classes)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    return np.array(all_true), np.array(all_preds)


def evaluate_model_with_loader(model, test_loader):
    """Evaluate model using a DataLoader (for disk-based data)."""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    return np.array(all_true), np.array(all_preds)


# Visualization functions
def plot_training_history(history):
    """Plot training and validation loss/accuracy curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('time_series_80x80_CNN_block_model_history.png') # TODO: UPDATE FIG NAME
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names_dict):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get unique classes in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_labels = [class_names_dict.get(int(cls), f"Class {cls}") for cls in unique_classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('time_series_80x80_CNN_block_model_conf_matrix.png') # TODO: UPDATE FIG NAME
    plt.show()
    
    # Also print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, 
                                  target_names=[class_names_dict.get(int(cls), f"Class {cls}") 
                                              for cls in unique_classes])
    print(report)


# Main execution block
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    s2_data_folder = os.path.join(project_root, 'data', 'raw', 'sentinel2_imagery')
    lc_data_folder = os.path.join(project_root, 'data', 'raw', 'USFS_land_cover')
    
    # Output directory for blocks
    block_data_dir = os.path.join(project_root, 'data', 'processed', 'time_series_blocks')
    os.makedirs(block_data_dir, exist_ok=True)
    
    roi_names = [
        "centennial", "cortez", "creede", "cripple_creek", "deer_tail",
        "durango", "gunnison", "hunter-fryingpan", "kit_carson", "lake_city"
    ]
    
    months = ['05', '06', '07', '08', '09']
    year = '2021'
    roi_list = generate_roi_list(roi_names, s2_data_folder, lc_data_folder, year, months)
    
    block_size = 64
    sample_size = 250000  
    
    # Generate training data and save to disk
    # dataset_info = generate_time_series_training_samples(
    #     roi_list, block_size, sample_size, output_dir=block_data_dir
    # )

    # Use existing blocks in memory
    print("Reading block metadata from memory")
    metadata_path = os.path.join(block_data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    dataset_info = {
        'metadata_path': metadata_path,
        'n_samples': len(metadata['blocks']),
        'n_classes': len(set(metadata['classes']))
    }
    print(f"Dataset prepared with {dataset_info['n_samples']} samples and {dataset_info['n_classes']} classes")
    
    # Train model using disk-based dataset
    # model, history = train_land_cover_model(
    #     dataset_info,
    #     batch_size=16,
    #     epochs=20,
    #     learning_rate=0.001,
    #     block_size=block_size
    # )
    
    # # Visualize training progress
    # plot_training_history(history)

    # MANUAL MODEL LOADING FOR EVALUATION
    checkpoint = torch.load('time_series_80x80_CNN_block_model.pth')

    # Initialize model with the saved parameters
    model = LandCoverCNN(
        num_classes=11,
        block_size=checkpoint['block_size']
    )
    
    # Evaluation - using disk-based dataset
    metadata_path = dataset_info['metadata_path']
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get indices for a test set (different from validation set)
    classes = np.array(metadata['classes'])
    train_val_indices, test_indices = train_test_split(
        np.arange(len(classes)), test_size=0.2, stratify=classes, random_state=24
    )
    
    # Create test dataset
    full_dataset = DiskLandCoverDataset(metadata_path, normalize=True)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate
    y_true, y_pred = evaluate_model_with_loader(model, test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    print("Evaluation complete!")



    