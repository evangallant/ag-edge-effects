import ee
import os
import geemap
import datetime
import time

def download_sentinel2_data(start_date, end_date, roi_list, cloud_percentage=20, bands=None):
    """
    Download Sentinel-2 data for multiple regions of interest
    
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        roi_list (list): List of dictionaries with 'name' and 'bounds' keys
                         'bounds' should be [min_lon, min_lat, max_lon, max_lat]
        cloud_percentage (float): Maximum cloud percentage allowed
        bands (list): List of Sentinel-2 bands to download (default: B2, B3, B4, B8, B11, B12)
        
    Returns:
        dict: Dictionary mapping ROI names to download paths or error messages
    """
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_folder = os.path.join(project_root, 'data', 'raw', 'sentinel2_imagery')
    os.makedirs(output_folder, exist_ok=True)
    
    # Default bands if none specified (10m and key 20m bands)
    if bands is None:
        bands = ['B2', 'B3', 'B4']
    
    date_str = f"{start_date}_to_{end_date}"
    results = {}
    
    for roi_item in roi_list:
        roi_name = roi_item['name']
        bounds = roi_item['bounds']
        
        # Create an EE geometry from the bounding box
        roi_geometry = ee.Geometry.Rectangle(bounds)
        
        try:
            # Filter the image collection
            print(f"Getting data from GEE for {roi_name}")
            dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(roi_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)) 
                # .map(mask_s2_clouds)
            
            # Check if we found any images
            image_count = dataset.size().getInfo()
            print(f"image count = {image_count}")

            if image_count == 0:
                print("no image found")
                results[roi_name] = f"No images found for {roi_name} with cloud percentage < {cloud_percentage}%"
                continue
                
            # Sort by cloud coverage and get the best image
            sorted_dataset = dataset.sort('CLOUDY_PIXEL_PERCENTAGE')
            best_image = sorted_dataset.first()
            
            # Select the desired bands and clip to ROI
            s2_image = best_image.select(bands).clip(roi_geometry)       
            file_name = f"sentinel2_10m_{roi_name.lower().replace(' ', '_')}_{start_date[0:7]}"
            output_path = os.path.join(output_folder, f"{file_name}.tif")

            # Export the image
            print(f"Downloading {file_name}...")
            try:
                # Direct export to file
                geemap.ee_export_image(
                    s2_image,
                    filename=output_path,
                    scale=10,
                    region=roi_geometry,
                    file_per_band=False
                )
                print(f"Successfully exported to {file_name}")
            except Exception as e:
                print(f"Error exporting {roi_name}: {str(e)}")
                return str(e)
            
            # Add to results
            results[roi_name] = file_name

            print(results)
            
            # Sleep to avoid hitting API limits
            time.sleep(2)
            
        except Exception as e:
            results[roi_name] = f"Error for {roi_name}: {str(e)}"
    
    return results

def mask_s2_clouds(image):
    """
    Mask clouds and cirrus clouds in Sentinel-2 imagery
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).neq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).neq(0))
    
    return image.updateMask(mask).divide(10000)


def download_USFS_data(start_date, end_date, roi_list):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_folder =  os.path.join(project_root, 'data', 'raw', 'USFS_land_cover')
    os.makedirs(output_folder, exist_ok=True)

    results = {}

    for roi_item in roi_list:
        roi_name = roi_item['name']
        bounds = roi_item['bounds']
        roi_geometry = ee.Geometry.Rectangle(bounds)
        
        # Make a unique filename for this ROI
        file_name = f"landcover_30m_{roi_name}_{start_date[0:7]}.tif"
        output_path = os.path.join(output_folder, file_name)

        try:
            # Get the USFS data
            usfs_data = ee.ImageCollection('USFS/GTAC/LCMS/v2023-9') \
                .filterBounds(roi_geometry) \
                .filter(ee.Filter.date(start_date, end_date)) \
                .first()
            
            # Check if we found any data
            if usfs_data is None:
                results[roi_name] = f"No USFS data found for {roi_name} in the specified date range"
                continue
                
            # Select the Land_Cover band and clip to ROI
            land_cover_data = usfs_data.select('Land_Cover').clip(roi_geometry)
            
            # Export the image
            print(f"Downloading USFS Land Cover data for {roi_name}...")
            geemap.ee_export_image(
                land_cover_data,
                filename=output_path,
                scale=30,
                region=roi_geometry,
                file_per_band=False
            )
            
            # Add to results
            results[roi_name] = output_path
            
            # Sleep to avoid hitting API limits
            time.sleep(2)
            
        except Exception as e:
            results[roi_name] = f"Error for {roi_name}: {str(e)}"
    
    return results


def download_categorical_upsampling_data(roi):
    # Get the USFS data for the whole country
    usfs_data = ee.ImageCollection('USFS/GTAC/LCMS/v2023-9') \
        .filterBounds(roi) \
        .filter(ee.Filter.date('2021', '2022')) \
        .first()

    land_cover_data = usfs_data.select('Land_Cover')

    # Get the Sentinel 2 data
    s2_data = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2021-09-01', '2022-09-30') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .first()

    s2band_data = s2_data.select(['B2', 'B3', 'B4', 'B8'])

    # Download the data to data/raw/bilinear_upsampling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_folder =  os.path.join(project_root, 'data', 'raw', 'categorical_upsampling')
    os.makedirs(output_folder, exist_ok=True)

    land_cover_file = os.path.join(output_folder, 'sentinel2_10m_NEON_region.tif')
    s2band_file = os.path.join(output_folder, 'sentinel2_10m_NEON_region.tif')

    try: 
        geemap.ee_export_image(land_cover_data, filename=land_cover_file, scale=30, region=roi)
        geemap.ee_export_image(s2band_data, filename=s2band_file, scale=10, region=roi)
    except Exception:
        return "Exception: " + Exception
    else:
        return "Files added to " + output_folder
    

if __name__ == "__main__":
    # Configuration
    roi_list = [
        {
            'name': 'kit_carson',
            'bounds': [-103.532941,38.620000,-103.361656,38.737928]
        },
        {
            'name': 'cripple_creek',
            'bounds': [-104.822874,38.306231,-104.655365,38.428297]
        },
        {
            'name': 'montrose',
            'bounds': [-108.010033,38.376833,-107.867239,38.484411]
        },
        {
            'name': 'cortez',
            'bounds': [-108.599581,37.365064,-108.676009,37.265412]
        },
        {
            'name': 'durango',
            'bounds': [-107.962482,37.263124,-107.852640,37.350509]
        },
        {
            'name': 'lizard_head',
            'bounds': [-108.122764,37.784825,-108.021161,37.857507]
        },
        {
            'name': 'ridgway',
            'bounds': [-107.817840,38.171273,-107.712117,38.250044]
        },
        {
            'name': 'uncompahgre',
            'bounds': [-108.716703,38.448570,-108.631575,38.515221]
        },
        {
            'name': 'yuma',
            'bounds': [-102.996995,39.883045,-102.903629,39.958877]
        },
        {
            'name': 'centennial',
            'bounds': [-104.852499,39.555589,-104.759134,39.627551]
        },
        {
            'name': 'gunnison',
            'bounds': [-107.037647,38.466492,-106.938790,38.539572]
        },
        {
            'name': 'powderhorn',
            'bounds': [-107.265907,38.147517,-107.157438,38.231708]
        },
        {
            'name': 'lake_city',
            'bounds': [-107.063429,37.678845,-106.953588,37.759986]
        },
        {
            'name': 'monte_vista',
            'bounds': [-105.953601,37.270498,-105.843759,37.361997]
        }
    ]

    # Initialize Earth Engine
    ee.Initialize(project='agedgeeffects')

    # download_sentinel2_data('2021-09-01', '2021-09-30', roi_list, cloud_percentage=50, bands=None)
    download_USFS_data('2021', '2022', roi_list)