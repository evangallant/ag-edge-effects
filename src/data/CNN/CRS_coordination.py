import rasterio
import os
import sys
import numpy as np
import tempfile
import traceback
from rasterio.warp import transform
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

cnn_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = Path(cnn_dir).parent
src_dir = Path(data_dir).parent
root_dir = Path(src_dir).parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src.data.CNN.cnn_data_generator import generate_training_samples

def visualize_sentinel2_file(tif_path, figsize=(24, 8), enhancement=1.0):
    """
    Visualize a single Sentinel-2 TIF file with diagnostics, supporting NIR band.
    
    Parameters:
    -----------
    tif_path : str
        Path to the Sentinel-2 TIF file
    figsize : tuple
        Figure size for the plot
    enhancement : float
        Contrast enhancement factor
    """
    try:
        with rasterio.open(tif_path) as src:
            # Get basic info
            print(f"File: {os.path.basename(tif_path)}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Bands: {src.count}")
            print(f"CRS: {src.crs}")
            
            # Read data
            data = src.read()
            
            # Print value ranges
            print(f"Value range - Min: {data.min()}, Max: {data.max()}")
            print(f"Mean: {data.mean()}, Median: {np.median(data)}")
            
            # Create figure with multiple components
            fig = plt.figure(figsize=figsize)
            
            # 1. RGB True Color Image (Bands 0,1,2 = B2,B3,B4)
            ax1 = fig.add_subplot(131)
            
            # Normalize RGB bands with percentile stretching
            rgb = np.zeros((3, src.height, src.width), dtype=np.float32)
            for i in range(3):  # Only the first 3 bands for RGB
                band = data[i].astype(np.float32)
                if band.max() > band.min():
                    # Get 2% and 98% percentile values
                    low = np.percentile(band[band > 0], 2)
                    high = np.percentile(band, 98)
                    
                    # Apply contrast stretch
                    band_norm = np.clip((band - low) / (high - low) * enhancement, 0, 1)
                    rgb[i] = band_norm
            
            # Create RGB composite
            rgb_composite = np.transpose(rgb, (1, 2, 0))
            
            # Display RGB image
            ax1.imshow(rgb_composite)
            ax1.set_title('RGB True Color (B2,B3,B4)')
            ax1.axis('off')
            
            # 2. False Color Composite (NIR,Red,Green = B8,B4,B3)
            ax2 = fig.add_subplot(132)
            
            # Check if we have NIR band (B8)
            if src.count >= 4:
                false_color = np.zeros((3, src.height, src.width), dtype=np.float32)
                
                # Order: NIR, Red, Green (B8,B4,B3)
                band_indices = [3, 2, 1]  # B8, B4, B3
                
                for i, band_idx in enumerate(band_indices):
                    band = data[band_idx].astype(np.float32)
                    if band.max() > band.min():
                        low = np.percentile(band[band > 0], 2)
                        high = np.percentile(band, 98)
                        band_norm = np.clip((band - low) / (high - low) * enhancement, 0, 1)
                        false_color[i] = band_norm
                
                # Create false color composite
                false_color_composite = np.transpose(false_color, (1, 2, 0))
                
                # Display false color image
                ax2.imshow(false_color_composite)
                ax2.set_title('False Color (NIR,Red,Green)')
            else:
                ax2.text(0.5, 0.5, 'NIR band not available', 
                         ha='center', va='center', fontsize=12)
            ax2.axis('off')
            
            # 3. NDVI (if NIR is available)
            ax3 = fig.add_subplot(133)
            
            if src.count >= 4:
                # Calculate NDVI
                nir = data[3].astype(np.float32)  # B8
                red = data[2].astype(np.float32)  # B4
                
                # Avoid division by zero
                ndvi = np.zeros_like(nir)
                valid = (nir + red) > 0
                ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
                
                # Display NDVI
                ndvi_plot = ax3.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                ax3.set_title('NDVI')
                plt.colorbar(ndvi_plot, ax=ax3, fraction=0.046, pad=0.04)
            else:
                ax3.text(0.5, 0.5, 'NIR band not available for NDVI', 
                         ha='center', va='center', fontsize=12)
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Additional diagnostic: check if mostly zeros
            zero_percentage = np.sum(data == 0) / data.size * 100
            print(f"Zero values: {zero_percentage:.2f}%")
            
            return data  # Return the data for additional inspection if needed
            
    except Exception as e:
        print(f"Error visualizing {tif_path}: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        return None


def visualize_multiple_tifs(tif_paths, rows=None, cols=None, figsize=(16, 12), vis_type='rgb'):
    """
    Visualize multiple Sentinel-2 TIF files in a grid layout.
    
    Parameters:
    -----------
    tif_paths : list
        List of paths to Sentinel-2 TIF files
    rows, cols : int
        Number of rows and columns in the grid
    figsize : tuple
        Figure size for the plot
    vis_type : str
        Visualization type: 'rgb', 'false_color', or 'ndvi'
    """
    n = len(tif_paths)
    print(f"Number of tifs to visualize: {n}")
    
    # Determine grid layout if not specified
    if rows is None and cols is None:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
    elif rows is None:
        rows = (n + cols - 1) // cols
    elif cols is None:
        cols = (n + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Make sure axes is always a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, tif_path in enumerate(tif_paths):
        if i >= rows * cols:
            break
            
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        try:
            with rasterio.open(tif_path) as src:
                # Read data
                data = src.read()
                
                if vis_type == 'rgb':
                    # RGB True Color (B2,B3,B4)
                    composite = np.zeros((3, src.height, src.width), dtype=np.float32)
                    for j in range(min(3, src.count)):
                        band = data[j].astype(np.float32)
                        if band.max() > band.min():
                            low = np.percentile(band[band > 0] if np.any(band > 0) else band, 2)
                            high = np.percentile(band, 98)
                            band_norm = np.clip((band - low) / (high - low), 0, 1)
                            composite[j] = band_norm
                    
                    title_suffix = 'RGB'
                    cmap = None
                
                elif vis_type == 'false_color' and src.count >= 4:
                    # False Color (NIR,Red,Green)
                    composite = np.zeros((3, src.height, src.width), dtype=np.float32)
                    band_indices = [3, 2, 1]  # B8, B4, B3
                    
                    for j, band_idx in enumerate(band_indices):
                        band = data[band_idx].astype(np.float32)
                        if band.max() > band.min():
                            low = np.percentile(band[band > 0] if np.any(band > 0) else band, 2)
                            high = np.percentile(band, 98)
                            band_norm = np.clip((band - low) / (high - low), 0, 1)
                            composite[j] = band_norm
                    
                    title_suffix = 'False Color'
                    cmap = None
                
                elif vis_type == 'ndvi' and src.count >= 4:
                    # NDVI
                    nir = data[3].astype(np.float32)  # B8
                    red = data[2].astype(np.float32)  # B4
                    
                    # Avoid division by zero
                    composite = np.zeros_like(nir)
                    valid = (nir + red) > 0
                    composite[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
                    
                    title_suffix = 'NDVI'
                    cmap = 'RdYlGn'
                
                else:
                    # If requested type isn't available, fall back to RGB
                    composite = np.zeros((3, src.height, src.width), dtype=np.float32)
                    for j in range(min(3, src.count)):
                        band = data[j].astype(np.float32)
                        if band.max() > band.min():
                            low = np.percentile(band[band > 0] if np.any(band > 0) else band, 2)
                            high = np.percentile(band, 98)
                            band_norm = np.clip((band - low) / (high - low), 0, 1)
                            composite[j] = band_norm
                    
                    title_suffix = 'RGB (fallback)'
                    cmap = None
                
                # Create final visualization
                if vis_type == 'ndvi':
                    # For NDVI, keep as 2D array with colormap
                    im = ax.imshow(composite, cmap=cmap, vmin=-1, vmax=1)
                else:
                    # For RGB/False color, transpose to (H,W,3)
                    composite = np.transpose(composite, (1, 2, 0))
                    im = ax.imshow(composite)
                
                # Add colorbar for NDVI
                if vis_type == 'ndvi':
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Display title and info
                ax.set_title(f"{os.path.basename(tif_path)}\n{title_suffix}", fontsize=10)
                ax.axis('off')
                
                # Print info below the subplot
                zero_percentage = np.sum(data == 0) / data.size * 100
                value_range = f"Range: {data.min():.4f}-{data.max():.4f}"
                ax.text(0.5, -0.1, f"{value_range}\nZeros: {zero_percentage:.1f}%", 
                        transform=ax.transAxes, ha='center', fontsize=8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=8, wrap=True)
            ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()