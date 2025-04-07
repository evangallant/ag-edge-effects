import os
import sys
import geemap.core as geemap
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
raw_data_dir = os.path.join(project_root, 'data', 'raw')
processed_data_dir = os.path.join(project_root, 'data', 'processed')

landcover_tif = raw_data_dir + '/categorical_upsampling/landcover_30m_NEON_region.tif'
s2band_tif = raw_data_dir + '/categorical_upsampling/sentinel2_10m_NEON_region.tif'
output_file = processed_data_dir + '/landcover_10m_categorical_upsample.tif'

def visualize_categorical_upsampling(Map):
    sys.path.append(os.path.abspath('..'))

    Map.add_raster(landcover_tif, colormap="tab10", layer_name="Orig Land Cover 30m")
    Map.add_raster(output_file, colormap="tab10", layer_name="Upscaled Land Cover 10m")

    Map.add_legend(title="Land Cover Classes", 
              legend_dict={
                  "Trees": "3179B8",    # land cover class 1
                  "Shrubs": "EE6493",   # land cover class 7
                  "Grass": "4FC1F5"     # land cover class 10
              })

    return Map

def categorical_upsampling(input_file, output_file, scale_factor=3):
    with rasterio.open(input_file) as src:
        # Read data and get unique classes
        data = src.read(1)
        classes = np.unique(data)
        
        # Calculate new dimensions
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)
        
        # Update transform for the new resolution
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Initialize probability array for each class
        prob_maps = np.zeros((len(classes), new_height, new_width), dtype=np.float32)
        
        # Create probability map for each class
        for i, class_value in enumerate(classes):
            # Create binary map (1 where class exists, 0 elsewhere)
            binary_map = (data == class_value).astype(np.float32)
            
            # Upscale using bilinear to get "soft" probability
            # This creates a gradient effect near boundaries
            upscaled_probs = ndimage.zoom(binary_map, scale_factor, order=1)
            
            # Store in probability array
            prob_maps[i] = upscaled_probs
        
        # Create final map by taking class with highest probability at each pixel
        result = np.zeros((new_height, new_width), dtype=data.dtype)
        for y in range(new_height):
            for x in range(new_width):
                class_index = np.argmax([prob_maps[i, y, x] for i in range(len(classes))])
                result[y, x] = classes[class_index]
        
        # Write result
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })
        
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            dst.write(result.reshape(1, new_height, new_width))
    
    return f"Categorically upscaled land cover saved to {output_file}"


categorical_upsampling(landcover_tif, output_file)