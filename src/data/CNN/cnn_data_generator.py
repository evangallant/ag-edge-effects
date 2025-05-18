import os
import sys
import rasterio
from tqdm import tqdm
import random
import json
import numpy as np
import rasterio.crs
from rasterio.warp import transform
from math import sqrt

def generate_time_series_training_samples(roi_list, block_size=80, sample_size=100000, output_dir="block_data"):
    """
    Generate training samples from time series and store on disk.
    """
    # Prepare metadata to store information about blocks
    metadata = {
        'blocks': [],
        'classes': [],
        'roi_names': [],
        'target_pixels': [],
        'utm_coords': [],
        'albers_coords': [],
        'lat_lon_coords': []
    }
    
    # Calculate samples per ROI
    roi_sample_size = int(sample_size/len(roi_list))

    # Process each ROI
    for i, roi_item in enumerate(roi_list):
        roi_name = roi_item['name']
        roi_dir = os.path.join(output_dir, f"roi_{roi_name}")
        os.makedirs(roi_dir, exist_ok=True)
        
        print(f"Generating training samples for {roi_name} ROI...")

        # Create list of image paths
        s2_tif_list = [
            roi_item['s2_path_1'],
            roi_item['s2_path_2'], 
            roi_item['s2_path_3'], 
            roi_item['s2_path_4'], 
            roi_item['s2_path_5']
        ]
        
        # Verify all files exist
        if not all(os.path.exists(path) for path in s2_tif_list):
            missing_files = [path for path in s2_tif_list if not os.path.exists(path)]
            print(f"Warning: Missing files for {roi_name}: {len(missing_files)} files")
            print(f"First missing file: {missing_files[0] if missing_files else 'None'}")
            continue

        # Generate target pixels from first image
        roi_target_pixels = generate_target_pixels(s2_tif_list[0], roi_sample_size, block_size)
        
        if len(roi_target_pixels) == 0:
            print(f"Warning: No valid target pixels generated for {roi_name}")
            continue

        # Generate time series blocks and save to disk
        block_paths, valid_targets = generate_time_series_blocks(
            roi_target_pixels, 
            s2_tif_list, 
            block_size, 
            output_dir=roi_dir
        )
        
        if len(block_paths) == 0:
            print(f"Warning: No valid time series blocks found for {roi_name}")
            continue
            
        print(f"Generated {len(block_paths)} valid time series blocks for {roi_name}")

        # Get the corresponding land cover classes 
        roi_classes, roi_utm_coords, roi_albers_coords, roi_lat_lon_coords = find_matching_lc_pixel_classes(
            roi_item['lc_path'], s2_tif_list[0], valid_targets
        )

        if len(roi_classes) > 0:
            # Add metadata for this ROI
            metadata['blocks'].extend(block_paths)
            metadata['classes'].extend(roi_classes.tolist() if isinstance(roi_classes, np.ndarray) else roi_classes)
            metadata['roi_names'].extend([roi_name] * len(block_paths))
            metadata['target_pixels'].extend([list(target) for target in valid_targets])
            
            # Add coordinate information
            metadata['utm_coords'].extend([list(coord) for coord in roi_utm_coords])
            metadata['albers_coords'].extend([list(coord) for coord in roi_albers_coords])
            metadata['lat_lon_coords'].extend([list(coord) for coord in roi_lat_lon_coords])
        else:
            print(f"Warning: No land cover classes extracted for {roi_name}")

    # Verify we have samples
    if len(metadata['blocks']) == 0:
        print("No blocks extracted from any ROIs")
        return "Failed to extract pixel blocks"
    
    # Save metadata to JSON file
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Saved metadata with {len(metadata['blocks'])} samples to {metadata_path}")
    
    # Return paths to the saved data
    return {
        'metadata_path': metadata_path,
        'n_samples': len(metadata['blocks']),
        'n_classes': len(set(metadata['classes'])),
        'output_dir': output_dir
    }
        
def generate_time_series_blocks(target_pixels, s2_tif_list, block_size=80, output_dir="block_data"):
    """
    Generate image blocks from multiple time points and save to disk instead of memory.
    
    Parameters:
    -----------
    target_pixels : list
        List of (row, col) tuples for target pixels
    s2_tif_list : list
        List of paths to Sentinel-2 TIF files
    block_size : int
        Size of blocks to extract
    output_dir : str
        Directory to save blocks
        
    Returns:
    --------
    block_paths : list
        List of paths to saved blocks
    valid_targets : list
        List of valid target pixels
    """
    radius = block_size // 2
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    blocks_dir = os.path.join(output_dir, "blocks")
    os.makedirs(blocks_dir, exist_ok=True)
    
    # Track valid targets and blocks by time
    valid_targets_by_time = {}
    
    # Process each time point to find valid targets
    for t, s2_tif in enumerate(s2_tif_list):
        print(f"Processing time point {t+1}/{len(s2_tif_list)}: {os.path.basename(s2_tif)}")
        
        with rasterio.open(s2_tif) as src:
            # Get dimensions only (no need to load all data yet)
            height, width = src.height, src.width
            valid_targets = []
            
            # Check which target pixels will be valid for this time point
            for row, col in target_pixels:
                # Calculate block boundaries
                row_start = row - radius
                row_end = row + radius + 1
                col_start = col - radius
                col_end = col + radius + 1
                
                # Check if the block is fully within the image
                if (0 <= row_start < row_end <= height and 
                    0 <= col_start < col_end <= width):
                    valid_targets.append((row, col))
            
            valid_targets_by_time[t] = valid_targets
            print(f"  Found {len(valid_targets)} valid blocks for time point {t+1}")
    
    # Find targets valid across all time points
    if len(s2_tif_list) > 0:
        common_targets = set(valid_targets_by_time[0])
        for t in range(1, len(s2_tif_list)):
            common_targets = common_targets.intersection(set(valid_targets_by_time[t]))
        
        common_targets = list(common_targets)
        print(f"Found {len(common_targets)} target pixels valid across all {len(s2_tif_list)} time points")
    else:
        common_targets = []
    
    # Now extract and save blocks only for common targets
    block_paths = []
    progress_bar = tqdm(total=len(common_targets), desc="Saving blocks to disk")
    
    for i, target in enumerate(common_targets):
        target_dir = os.path.join(blocks_dir, f"block_{i:05d}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Store block information for each time point
        block_data_for_target = []
        
        for t, s2_tif in enumerate(s2_tif_list):
            with rasterio.open(s2_tif) as src:
                row, col = target
                data = src.read()
                n_bands, height, width = data.shape
                
                # Extract individual bands
                blue_band = data[0]    # B2
                green_band = data[1]   # B3
                red_band = data[2]     # B4
                nir_band = data[3]     # B8
                
                # Calculate indices
                indices = []
                
                # 1. NDVI
                ndvi = np.zeros_like(red_band, dtype=np.float32)
                denominator = nir_band + red_band
                valid_mask = denominator > 0
                ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
                indices.append(ndvi.reshape(1, height, width))
                
                # 2. EVI
                evi = np.zeros_like(red_band, dtype=np.float32)
                denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
                valid_mask = denominator != 0
                evi[valid_mask] = 2.5 * (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
                indices.append(evi.reshape(1, height, width))
                
                # 3. NDWI
                ndwi = np.zeros_like(red_band, dtype=np.float32)
                denominator = green_band + nir_band
                valid_mask = denominator > 0
                ndwi[valid_mask] = (green_band[valid_mask] - nir_band[valid_mask]) / denominator[valid_mask]
                indices.append(ndwi.reshape(1, height, width))
                
                # Add SWIR indices if available
                if n_bands >= 6:
                    swir1_band = data[4]   # B11
                    swir2_band = data[5]   # B12
                    
                    # 4. NDMI
                    ndmi = np.zeros_like(red_band, dtype=np.float32)
                    denominator = nir_band + swir1_band
                    valid_mask = denominator > 0
                    ndmi[valid_mask] = (nir_band[valid_mask] - swir1_band[valid_mask]) / denominator[valid_mask]
                    indices.append(ndmi.reshape(1, height, width))
                    
                    # 5. NBR
                    nbr = np.zeros_like(red_band, dtype=np.float32)
                    denominator = nir_band + swir2_band
                    valid_mask = denominator > 0
                    nbr[valid_mask] = (nir_band[valid_mask] - swir2_band[valid_mask]) / denominator[valid_mask]
                    indices.append(nbr.reshape(1, height, width))
                
                # Stack original bands with indices
                enhanced_data = np.vstack([data] + indices)
                
                # Extract block
                row_start = row - radius
                row_end = row + radius + 1
                col_start = col - radius
                col_end = col + radius + 1
                
                block = enhanced_data[:, row_start:row_end, col_start:col_end]
                
                # Save block for this time point
                time_path = os.path.join(target_dir, f"time_{t}.npy")
                np.save(time_path, block)
                block_data_for_target.append(time_path)
        
        # Save path to this block
        block_path = target_dir
        block_paths.append(block_path)
        progress_bar.update(1)
    
    progress_bar.close()
    return block_paths, common_targets

# Function to create training samples
def generate_training_samples(roi_list, block_size=80, sample_size=1000):
    """
        Generates training sample pairs - USFS land use type at target pixel/s2 bands BSxBS pixels around target pixel
        
        Parameters:
        -----------
            roi_list: a dictionary of roi names and file paths
            block_size: int - the dimensions of the pixel box surrounding target pixels (larger = more context)
            sample_count: the number of samples to generate

        Returns:
        -----------
            s2_blocks: an array of the s2 band patches surrounding the target pixels
            classes: an array of the USFS land cover classes for the target pixels
            s2_block_metadata: dictionary of metadata for each target pixel, containing the origin ROI, file path, and target pixel value
            utm_coords: list of tuples representing the UTM coordinates of each target pixel
            albers_coords: list of tuples representing the Albers coordinates of each target pixel
    """
    # Initialize arrays
    classes = []
    s2_blocks = []
    s2_block_metadata = []
    utm_coords = []     
    albers_coords = []  
    lat_lon_coords = []

    # Sample size for each roi (evenly distributed)
    roi_sample_size = int(sample_size/len(roi_list))

    # Generate target pixels from each roi
    for roi_item in roi_list:
        roi_name = roi_item['name']
        print("Generating training samples for ", roi_name, " roi...")
        roi_s2_path = roi_item['s2_path']
        roi_lc_path = roi_item['lc_path']

        # Pick roi_sample_size number of target pixels randomly from the s2 data from the current roi
        roi_target_pixels = generate_target_pixels(roi_s2_path, roi_sample_size, block_size)

        # Get the BSxBS s2 pixel boxes with the target pixels at the center, and save them to the s2pixels array
        # Here we don't need to do any coordinate finagling, since the target pixel coords come directly from the sentinel 2 images 
        # valid_target_pixels gives us the list of all target pixels we were able to extract a pixel block for
        roi_s2_blocks, valid_target_pixels = generate_s2_pixel_blocks(roi_s2_path, roi_target_pixels, block_size)

        if len(roi_s2_blocks) > 0:
            s2_blocks.extend(roi_s2_blocks)
            for i in range(len(roi_s2_blocks)):
                s2_block_metadata.append({
                    'roi': roi_name,
                    'file_path': roi_s2_path,
                    'target_pixel': valid_target_pixels[i]
                })

        # Get the corresponding USFS land cover classes for the target pixels, and save them in the classes array
        roi_classes, roi_utm_coords, roi_albers_coords, roi_lat_lon_coords = find_matching_lc_pixel_classes(roi_lc_path, roi_s2_path, valid_target_pixels)

        if len(roi_classes) > 0:
            classes.extend(roi_classes)
            utm_coords.extend(roi_utm_coords)        
            albers_coords.extend(roi_albers_coords)  
            lat_lon_coords.extend(roi_lat_lon_coords)


    # Convert list of pixel blocks to a single NumPy array
    if s2_blocks:
        s2_blocks = np.stack(s2_blocks)
    else:
        print("No blocks extracted from any ROIs")
        return "Failed to extract pixel blocks"
    
    # Convert list of classes for target pixels to a single NumPy array
    if classes:
        classes = np.stack(classes)
    else:
        print("No land cover classes extracted for target pixels")
        return "Failed to extract land cover classes"
    
    return s2_blocks, classes, s2_block_metadata, utm_coords, albers_coords, lat_lon_coords

def generate_target_pixels(s2_tif, roi_sample_size, block_size=80):
    # Generates roi_sample_size number of random target pixels for an roi
    target_pixels = []

    # Open the s2 tif
    with rasterio.open(s2_tif) as src:
        b1 = src.read(1)
        rows, cols = b1.shape
        buffer = int(block_size * 3)

        # Generate random target pixels
        for i in range(roi_sample_size):
            row = random.randint(buffer, (rows - buffer))
            col = random.randint(buffer, (cols - buffer))

            if b1[row, col] != src.nodata:
                target_pixels.append((row, col))

    return target_pixels

def generate_s2_pixel_blocks(s2_tif, target_pixels, block_size=80):
    # Given a set of target pixels, return a list of arrays which contain the BSxBS pixel blocks with all 3 RGB bands
    radius = block_size // 2

    with rasterio.open(s2_tif) as src:
        # For each target pixel, extract the surrounding pixels and store them as an item in the pixel_blocks array
        data = src.read() # Read all bands
        n_bands, height, width = data.shape

        # Initialize lists to store blocks and valid targets
        blocks = []
        valid_targets = []

        # For each target pixel
        for row, col in target_pixels:
            # Calculate block boundaries
            row_start = row - radius
            row_end = row + radius + 1  # +1 because slicing is exclusive
            col_start = col - radius
            col_end = col + radius + 1

            # Check if the block is fully within the image
            if (0 <= row_start < row_end <= height and 
                0 <= col_start < col_end <= width):
                # Extract all bands for this block
                block = data[:, row_start:row_end, col_start:col_end] 
                blocks.append(block)
                valid_targets.append((row, col))
        
    return blocks, valid_targets

def find_matching_lc_pixel_classes(lc_tif, s2_tif, valid_target_pixels):
    """
        Given a list of target pixels from the s2 data, return a list of ints for the land use classes from the corresponding lc pixels at the same locations.

        Parameters:
            lc_tif: tif file path for USFS 30m land cover data
            s2_tif: tif file path for sentinel 2 10m data
            valid_target_pixels: array of (row, col) tuples representing target pixels from sentinel 2 imagery

        Returns:
            roi_classes: list of ints representing the land cover class from USFS data the location of each target pixel
            utm_coords: list of tuples representing the UTM coordinates of each target pixel
            albers_coords: list of tuples representing the Albers coordinates of each target pixel
    """
    roi_classes = []
    utm_target_xs = []
    utm_target_ys = []
    utm_coords = []
    albers_coords = []
    lat_lon_coords = []

    with rasterio.open(s2_tif) as s2_src:
        s2_crs = s2_src.crs

        with rasterio.open(lc_tif) as lc_src:
            lc_crs = lc_src.crs
            lc_data = lc_src.read(1)

            for valid_pixel in valid_target_pixels:
                x = valid_pixel[0]
                y = valid_pixel[1]

                # Real world coordinates:
                utm_x, utm_y = s2_src.xy(x, y)

                utm_target_xs.append(float(utm_x))
                utm_target_ys.append(float(utm_y))

            # Transform to Albers projection
            albers_xs, albers_ys = rasterio.warp.transform(s2_crs, lc_crs, utm_target_xs, utm_target_ys)
            # Transform to lat/long (ESPG:4326) in case it's needed
            lat_lon_crs = rasterio.CRS.from_epsg(4326)
            lats, lons = rasterio.warp.transform(s2_crs, lat_lon_crs, utm_target_xs, utm_target_ys)
            
            for i in range(len(albers_xs)):
                row, col = lc_src.index(albers_xs[i], albers_ys[i])

                # Get land cover class at that point from the 
                pixel_class = lc_data[row, col]
                roi_classes.append(int(pixel_class))

    if len(utm_target_xs) != len(utm_target_ys) != len(albers_xs) != len(albers_ys) != len(lat_lon_coords):
        return "Lost row and/or column coordinates during transformation"
    else:
        for i in range(len(utm_target_xs)):
            utm_coords.append((utm_target_xs[i], (utm_target_ys[i])))
            albers_coords.append((albers_xs[i], albers_ys[i]))
            lat_lon_coords.append((lats[i], lons[i]))

    return roi_classes, utm_coords, albers_coords, lat_lon_coords


def generate_roi_list(location_names, s2_data_folder, lc_data_folder, year="2021", months=None):
    """
    Generate a list of ROI dictionaries based on location names.
    
    Parameters:
    -----------
    location_names : list of str
        List of location names (e.g., ["Grand Junction", "La Junta"])
    s2_data_folder : str
        Path to the folder containing Sentinel-2 imagery
    lc_data_folder : str
        Path to the folder containing land cover data
    year : str, optional
        Year for the data (default: "2021")
    months : list of str, optional
        List of months for time series (e.g., ["05", "06", "07", "08", "09"])
        If None, will use single-image approach
        
    Returns:
    --------
    roi_list : list of dict
        List of dictionaries with paths for each location
    """
    roi_list = []

    if months is None:
        # Single-image approach
        default_month = "07"
        for location_name in location_names:
            location_slug = location_name.lower().replace(" ", "_")
            
            lc_path = os.path.join(lc_data_folder, f"landcover_30m_{location_slug}_{year}.tif")
            s2_path = os.path.join(s2_data_folder, 'rgb_nir_tifs', f"sentinel2_10m_{location_slug}_{year}-{default_month}.tif")
            
            if not os.path.exists(lc_path):
                print(f"Warning: Land cover file not found: {lc_path}")
                continue
                
            if not os.path.exists(s2_path):
                print(f"Warning: Sentinel-2 file not found: {s2_path}")
                continue
            
            roi_list.append({
                'name': location_name,
                's2_path': s2_path,
                'lc_path': lc_path
            })
    else:
        # Time series approach
        for location_name in location_names:
            location_slug = location_name.lower().replace(" ", "_")
            
            # Check land cover file
            lc_path = os.path.join(lc_data_folder, f"landcover_30m_{location_slug}_{year}.tif")
            if not os.path.exists(lc_path):
                print(f"Warning: Land cover file not found: {lc_path}")
                continue
            
            # Check all S2 files exist
            s2_paths = []
            all_exist = True
            
            for i, month in enumerate(months):
                s2_path = os.path.join(s2_data_folder, f"sentinel2_10m_{location_slug}_{year}-{month}.tif")
                s2_paths.append(s2_path)
                
                if not os.path.exists(s2_path):
                    print(f"Warning: Sentinel-2 file not found: {s2_path}")
                    all_exist = False
                    break
            
            # Only add ROI if all files exist
            if all_exist:
                roi_dict = {'name': location_name, 'lc_path': lc_path}
                
                # Add numbered paths
                for i, path in enumerate(s2_paths):
                    roi_dict[f's2_path_{i+1}'] = path
                
                roi_list.append(roi_dict)
            else:
                print(f"Skipping {location_name} due to missing files")

    return roi_list


if __name__ == "__main__":
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    s2_data_folder = os.path.join(project_root, 'data', 'raw', 'sentinel2_imagery')
    lc_data_folder = os.path.join(project_root, 'data', 'raw', 'USFS_land_cover')
    
    location_names = [
        "grand_junction", 
        "la_junta", 
        "pawnee_grasslands", 
        "san_luis_valley",
        "Walden",
        "Saguache",
        "Kit_Carson",
        "Cripple_Creek",
        # "Montrose",
        # "Cortez",
        # "Durango",
        # "Lizard_Head",
        # "Ridgway",
        # "Uncompahgre",
        # "Yuma",
        # "Centennial",
        # "Gunnison",
        # "Powderhorn",
        # "Lake_City",
        "Monte_Vista"
    ]

    # Generate the ROI list
    months = ['05', '06',' 07', '08', '09']
    year = '2021'
    roi_list = generate_roi_list(location_names, s2_data_folder, lc_data_folder, year, months)

    print(f"Generated ROI list with {len(roi_list)} valid locations:")
    for roi in roi_list:
        print(f"  - {roi['name']}")
    
    block_size = 64
    sample_size = 100
    
    print(f"Testing configuration:")
    print(f"Block Size: {block_size}")
    print(f"Sample Size: {sample_size}")
    print(f"Number of ROIs: {len(roi_list)}")
    
    # Run the analysis
    s2_blocks, classes, s2_block_metadata, utm_coords, albers_coords, latslons = generate_training_samples(roi_list, block_size, sample_size)
    classes_found, class_distribution = np.unique(classes, return_counts=True)
    
    # Optional: Add validation or summary
    print(f"Successfully generated {len(s2_blocks)} training samples ({((len(s2_blocks)/sample_size)*100):.2f}% of requested sample size)")
    for i, lc in enumerate(classes_found):
        print(f"Found {class_distribution[i]} pixels of class {classes_found[i]}")

    print(f"utm coords list: {utm_coords}")
    print(f"albers coords list: {albers_coords}")
    print(f"lat lon list: {latslons}")