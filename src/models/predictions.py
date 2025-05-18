import torch
import numpy as np
import rasterio
import os
import sys
import folium
from pathlib import Path
from src.models.cnn_model import LandCoverCNN  # Import your model definition

def load_model(model_path, num_classes=15):
    # Load the state dictionary first to inspect
    checkpoint = torch.load(model_path)

    # Initialize model with the saved parameters
    model = LandCoverCNN(
        num_classes=num_classes,
        block_size=checkpoint['block_size']
    )

    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def extract_block(s2_tif, row, col, block_size=15):
    """Extract a block from the Sentinel-2 image"""
    with rasterio.open(s2_tif) as src:
        radius = block_size // 2
        data = src.read()
        block = data[:, row-radius:row+radius+1, col-radius:col+radius+1]
        return block

def predict_land_cover(model, image_block):
    model.eval()
    device = next(model.parameters()).device
    
    # Preprocess - ensure correct shape and normalization
    if isinstance(image_block, np.ndarray):
        # Normalize if needed
        if image_block.max() > 1.0:
            image_block = image_block.astype(np.float32) / 10000.0
            
        # Add batch dimension if needed
        if len(image_block.shape) == 3:  # (C, H, W)
            image = torch.tensor(image_block, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            image = torch.tensor(image_block, dtype=torch.float32).to(device)
    else:
        # Already a tensor
        image = image_block.unsqueeze(0).to(device) if len(image_block.shape) == 3 else image_block.to(device)
    
    # NOTE: We don't need to calculate NDVI here - the model's forward() method does that internally
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    # Add 1 to convert from model output (0-14) back to USFS classes (1-15)
    return predicted.item() + 1


# if __name__ == "__main__":
#     # Given an s2 tif

#     # Load model
#     model = load_model("[MODEL FILE PATH HERE TODO]")
    
#     # Extract block
#     block = extract_block(args.s2_tif, args.row, args.col)
    
#     # Predict
#     predicted_class = predict_land_cover(model, block)
    
#     # Print result
#     print(f"Predicted class: {predicted_class}")