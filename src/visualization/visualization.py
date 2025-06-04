"""
Given an ROI:
    1) Predicts the land cover class at each pixel
    2) Initializes an interactive Geemap map
    3) Creates a map layer of those predictions
    4) Creates a map layer of USFS data
    5) Creates a map layer comparing predictions to USFS data

Here we use the rgb_nir_80x80_CNN_block_model in the following steps:
    1) Load the ROI tif
    2) Initialize the output layer - 200 px smaller than the .tif file, so the pixel block approach works
    3) for each pixel in the output layer, set it's value to the prediction made by the model
    4) Get the USFS values layer
    5) Create a 'differences' layer that compares predictions to outputs
    6) Create a Geemap map with all 3 layers
"""
import sys
import os
import ee
import geemap
import torch
import numpy as np
import numpy.ma as ma
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from tqdm.notebook import tqdm

# Get helpers
script_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = Path(script_dir).parent
from src.models.predictions import predict_land_cover
from src.models.cnn_model import LandCoverCNN
from src.data.CNN.cnn_data_generator import generate_s2_pixel_blocks, find_matching_lc_pixel_classes

def get_blocks_and_classes(s2_tif, lc_tif=None):
    radius = 40
    with rasterio.open(s2_tif) as src:
        # For each target pixel, extract the surrounding pixels and store them as an item in the pixel_blocks array
        data = src.read() # Read all bands
        n_bands, height, width = data.shape

        # Initialize lists to store blocks and valid targets
        blocks = []
        valid_targets = []

        print(f"Retrieving pixel blocks from s2 tif of size {height} x {width}, with a total of {height * width} pixels")
        for row in range(450, (height - 450)):
            if (row + 1) % 200 == 0:
                print(f"Processing {row}th row / {height}")
            for col in range(450, (width - 450)):               
                row_start = row - radius
                row_end = row + radius + 1  # +1 because slicing is exclusive
                col_start = col - radius
                col_end = col + radius + 1

                block = data[:, row_start:row_end, col_start:col_end] 
                blocks.append(block)
                valid_targets.append((row, col))
    if lc_tif == None:
        usfs_classes, utms, albers, lat_lons = [], [], [], []
    else:
        print("Retrieving USFS land use classes")
        usfs_classes, utms, albers, lat_lons = find_matching_lc_pixel_classes(lc_tif, s2_tif, valid_targets)
        
    return blocks, valid_targets, usfs_classes, utms, albers, lat_lons

def get_pixel_coords(lat, lon, transform):
    """Convert geographic coordinates to pixel coordinates"""
    # This might need adjustment based on your coordinate system
    x, y = lon, lat  # For simple lat/lon (EPSG:4326)
    
    # Using rasterio's transform to convert to pixel coordinates
    row, col = ~transform * (x, y)
    
    # Round to integers for array indexing
    return int(row), int(col)

def visualize_tif(roi, tif_layers=False, usfs=True):
    """
        Given a valid ROI, returns a GeeMap object with predictions and comparisons
        roi: string - the roi name
        tif_layers: list of .tif file paths, if extant already
    """
    # Get data paths
    s2_data_folder = os.path.join(root_dir, 'data', 'raw', 'sentinel2_imagery', 'rgb_nir_tifs')
    lc_data_folder = os.path.join(root_dir, 'data', 'raw', 'USFS_land_cover')
    visuals_folder = os.path.join(root_dir, 'src', 'visualization')
    rgb_model_path = os.path.join(root_dir, 'src', 'models', 'rgb_nir_80x80_CNN_block_model.pth')
    s2_tif = Path(s2_data_folder) / f'sentinel2_10m_{roi}_2021-07.tif'
    if usfs == True:
        lc_tif = Path(lc_data_folder) / f'landcover_30m_{roi}_2021.tif'

    ee.Initialize(project='agedgeeffects')

    if tif_layers == False:
        # Generate all valid s2 blocks, coords, and corresponding USFS classes from the ROI
        if usfs == True:
            blocks, valid_targets, usfs_classes, utms, albers, lat_lons = get_blocks_and_classes(s2_tif, lc_tif)
        else:
            blocks, valid_targets, usfs_classes, utms, albers, lat_lons = get_blocks_and_classes(s2_tif)

        # Load the model
        checkpoint = torch.load(rgb_model_path)
        model = LandCoverCNN(
            num_classes=15,
            block_size=checkpoint['block_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Initialize the output tif files
        print("Initializing output tif files")
        with rasterio.open(s2_tif) as src:
            s2_meta = src.meta.copy()
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs

            # Create empty numpy arrays the same dimensions as the s2 tif for:
                # predictions 
                # comparisons
                # usfs classes
                # a mask describing where predictions were able to be made
            prediction_array = np.zeros((height, width), dtype=np.uint8)
            if usfs == True:
                comparison_array = np.zeros((height, width), dtype=np.uint8)
                usfs_class_array = np.zeros((height, width), dtype=np.uint8)
            valid_pixel_mask = np.zeros((height, width), dtype=bool)


        ############# Populate the output tif files
        # Function to get the pixel coordinates given lat/lons
        print(f"Number of blocks to predict: {len(blocks)}")
        dif = 0
        for i in tqdm(range(len(blocks)), desc="Processing pixel blocks", unit="block"):
            block = blocks[i]
            if usfs == True:
                usfs_class = usfs_classes[i]
            row, col = valid_targets[i]
            # row, col = get_pixel_coords(lon, lat, transform)
            
            # Add the predicted class to the prediction array and mark it as valid in the mask
            predicted_class = predict_land_cover(model, block)
            prediction_array[row, col] = predicted_class
            valid_pixel_mask[row, col] = True
            if usfs == True:
                usfs_class_array[row, col] = usfs_class

            # Update the comparison array
            if usfs == True:
                if predicted_class == usfs_class:
                    comparison_array[row, col] = 1
                else:
                    comparison_array[row, col] = 2
                    dif += 1

        print(f"{round((dif/len(blocks))*100), 2}% of pixels predicted differently.")

        # Apply the valid pixel mask to both the prediction and comparison arrays
        if usfs == True:
            prediction_array = ma.array(prediction_array, mask=~valid_pixel_mask)
            usfs_class_array = ma.array(usfs_class_array, mask=~valid_pixel_mask)
            comparison_array = ma.array(comparison_array, mask=~valid_pixel_mask)

        # Now that we have our output tifs, write them to memory for viz
        # First, create the metadata for the prediction tif
        print("Writing to .tif files")
        prediction_meta = s2_meta.copy()
        prediction_meta.update({
            'count': 1, # Single band of predictions
            'dtype': 'uint8',
            'nodata': 0
        })

        # Write the prediction layer
        prediction_tif_path = os.path.join(visuals_folder, f'{roi}_prediction_layer.tif')
        with rasterio.open(prediction_tif_path, 'w', **prediction_meta) as dst:
            dst.write(prediction_array, 1)

        # usfs classes tif
        if usfs == True:
            usfs_meta = s2_meta.copy()
            usfs_meta.update({
                'count': 1, # Single band of predictions
                'dtype': 'uint8',
                'nodata': 0
            })

            # Write the prediction layer
            usfs_class_tif_path = os.path.join(visuals_folder, f'{roi}_usfs_class_layer.tif')
            with rasterio.open(usfs_class_tif_path, 'w', **usfs_meta) as dst:
                dst.write(usfs_class_array, 1)

            # Now create the metadata for the classes tif
            comparison_meta = s2_meta.copy()
            comparison_meta.update({
                'count': 1,
                'dtype': 'uint8',
                'nodata': 0
            })

            # Write the comparison layer
            comparison_tif_path = os.path.join(visuals_folder, f'{roi}_comparison_layer.tif')
            with rasterio.open(comparison_tif_path, 'w', **comparison_meta) as dst:
                dst.write(comparison_array, 1)

        print("Output tifs created and saved to memory, creating GeeMap visualization")

    prediction_tif = os.path.join(visuals_folder, f'{roi}_prediction_layer.tif')
    if usfs == True:
        usfs_class_tif = os.path.join(visuals_folder, f'{roi}_usfs_class_layer.tif')
        comparison_tif = os.path.join(visuals_folder, f'{roi}_comparison_layer.tif')

    ############# Create the visualizations
    class_legend = {
        "Trees": "#1f77b4",                            # Blue
        "Tall Shrubs & Trees Mix (SEAK Only)": "#aec7e8", # Light blue
        "Shrubs & Trees Mix": "#ff7f0e",               # Orange
        "Grass/Forb/Herb & Trees Mix": "#ffbb78",      # Light orange
        "Barren & Trees Mix": "#2b9f2b",               # Green
        "Tall Shrubs (SEAK Only)": "#a5e298",          # Light green
        "Shrubs": "#d62728",                           # Red
        "Grass/Forb/Herb & Shrubs Mix": "#ff9896",     # Light red
        "Barren & Shrubs Mix": "#9467bd",              # Purple
        "Grass/Forb/Herb": "#c5b0d5",                  # Light purple
        "Barren & Grass/Forb/Herb Mix": "#8c564b",     # Brown
        "Barren or Impervious": "#c49c94",             # Light brown
        "Snow or Ice": "#e376c2",                      # Pink
        "Water": "#f7b6d2",                            # Cyan
        "Non-Processing Area Mask": "#7f7f7f"          # Olive
    }

    Map = geemap.Map()
    Map.setCenter(38.476, -107.911, 15)
    Map.add_basemap("HYBRID")
    Map.add_raster(prediction_tif, colormap="tab20", layer_name="Predictions")
    if usfs == True:
        Map.add_raster(usfs_class_tif, colormap="tab20", layer_name="USFS Classes")
        Map.add_raster(comparison_tif, colormap="binary", layer_name="Comparison")
    Map.add_legend(title="Land Cover Classes", 
               legend_dict=class_legend)
    Map.add_layer_control()

    return Map