import rasterio
import os
import sys
import numpy as np
import folium
import tempfile
from folium.plugins import MeasureControl
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

def visualize_sentinel2_file(tif_path, figsize=(24, 20), enhancement=1.0):
    """
    Visualize a single Sentinel-2 TIF file with diagnostics.
    
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
            
            # 1. RGB True Color Image
            ax1 = fig.add_subplot(131)
            
            # Normalize RGB bands with percentile stretching
            rgb = np.zeros((3, src.height, src.width), dtype=np.float32)
            for i in range(min(3, src.count)):
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
            ax1.set_title('RGB True Color')
            ax1.axis('off')
            
            # 2. Histogram of values by band
            # ax2 = fig.add_subplot(132)
            # colors = ['blue', 'green', 'red']
            # for i in range(min(3, src.count)):
            #     band = data[i].flatten()
            #     band = band[band > 0]  # Remove zeros for better histogram
            #     if len(band) > 0:
            #         ax2.hist(band, bins=50, alpha=0.5, label=f'Band {i+1}', color=colors[i])
            
            # ax2.set_title('Histogram (non-zero values)')
            # ax2.set_xlabel('Pixel Value')
            # ax2.set_ylabel('Frequency')
            # ax2.legend()
            
            # 3. First Band as Grayscale (to check if data exists)
            # ax3 = fig.add_subplot(133)
            # band1 = data[0].astype(np.float32)
            # if band1.max() > band1.min():
            #     band1_norm = (band1 - band1.min()) / (band1.max() - band1.min())
            #     ax3.imshow(band1_norm, cmap='gray')
            # else:
            #     ax3.text(0.5, 0.5, 'No Data', ha='center', va='center')
            
            # ax3.set_title('Band 1 (Blue) Grayscale')
            # ax3.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Additional diagnostic: check if mostly zeros
            zero_percentage = np.sum(data == 0) / data.size * 100
            print(f"Zero values: {zero_percentage:.2f}%")
            
            return data  # Return the data for additional inspection if needed
            
    except Exception as e:
        print(f"Error visualizing {tif_path}: {str(e)}")
        return None
    

def visualize_multiple_tifs(tif_paths, rows=None, cols=None, figsize=(16, 12)):
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

                # Create RGB composite with enhancement
                rgb = np.zeros((3, src.height, src.width), dtype=np.float32)
                for j in range(min(3, src.count)):
                    band = data[j].astype(np.float32)
                    if band.max() > band.min():
                        # Get 2% and 98% percentile values for better contrast
                        low = np.percentile(band[band > 0] if np.any(band > 0) else band, 2)
                        high = np.percentile(band, 98)
                        
                        # Apply contrast stretch with enhancement
                        band_norm = np.clip((band - low) / (high - low) * 1.0, 0, 1)
                        rgb[j] = band_norm
                
                # Create RGB composite
                rgb_composite = np.transpose(rgb, (1, 2, 0))
                
                # Display RGB image
                ax.imshow(rgb_composite)
                ax.set_title(os.path.basename(tif_path), fontsize=10)
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