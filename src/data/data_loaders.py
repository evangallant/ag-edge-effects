import ee
import geemap.core as geemap

# Initialize Earth Engine
ee.Initialize(project='agedgeeffects')

def return_sentinel_data(map, start_date, end_date, cloud_percentage, bands, roi):
    """Returns a geemap object of the designated area.

    Args:
        map: a geemap Map object
        start_date: YYYY-MM-DD formatted string
        end_date: YYYY-MM-DD formatted string
        cloud_percentage: 0-100 filter value to get less cloudy images - int
        bands: an array of band values, length 1-3
        roi: region of interest in the form of an ee.Geometry.BBox object

    Returns:
        Geemap layer object of sentinel2 band data
    """
    Map = map

    # Function to mask clouds found in images
    def mask_s2_clouds(image):
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask).divide(10000)
    

    if len(bands) < 1 or len(bands) > 3:
       return "Please provide between 1 and 3 bands"
    
    # Get the image collection per our query variables
    dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)) \
        .map(mask_s2_clouds)

    visualization = {
        'min': 0.0,
        'max': 0.3,
        'bands': bands,
    }

    # Find the first image in the collection and clip it to our ROI
    if dataset.size().getInfo() > 0:
        sorted_dataset = dataset.sort('CLOUDY_PIXEL_PERCENTAGE')
        best_image = sorted_dataset.first()
        clipped_image = best_image.clip(roi)

    # Add the clipped image and ROI boundary to the map
    Map.add_layer(clipped_image, visualization, 'Sentinel Data')
    Map.addLayer(ee.FeatureCollection([ee.Feature(roi)]), {}, 'ROI', False)   # ROI checking layer

    # Return the map 
    return Map


def return_usda_data(map, start_date, end_date, roi):
    """Returns USDA crop type and coverage data in a geemap layer object

    Args:
        map: a geemap Map object
        start_date: YYYY-MM-DD formatted string
        end_date: YYYY-MM-DD formatted string
        roi: region of interest in the form of an ee.Geometry.BBox object

    Returns:
        Geemap layer object of USDA Crop cover data
    """
    Map = map
    
    # Initialize Earth Engine
    ee.Initialize(project='agedgeeffects')

    # Load the dataset
    dataset = ee.ImageCollection('USDA/NASS/CDL') \
        .filterBounds(roi) \
        .filter(ee.Filter.date(start_date, end_date)) \
        .first()
        
    # Get the crop type layer
    cropLandcover = dataset.select('cropland')

    # Clip the crop type layer to our ROI
    if cropLandcover:
        clipped_image = cropLandcover.clip(roi)

    legend = {
        'Corn': '#ffd400',
        'Soybeans': '#267300',
    }

    Map.addLayer(clipped_image, {}, 'Crop Landcover', opacity=0.35)
    Map._add_legend(legend_title='Crop Type', legend_dict=legend, postion='bottomright')

    # Return the map
    return Map


def return_usfs_data(map, start_date, end_date, roi):
    """Returns USFS land cover data in a geemap layer object

    Args:
        map: a geemap Map object
        start_date: YYYY-MM-DD formatted string
        end_date: YYYY-MM-DD formatted string
        roi: region of interest in the form of an ee.Geometry.BBox object

    Returns:
        Geemap layer object of USFS land cover data
    """
    Map = map
    
    # Initialize Earth Engine
    ee.Initialize(project='agedgeeffects')

    # Load the dataset
    dataset = ee.ImageCollection('USFS/GTAC/LCMS/v2023-9') \
        .filterBounds(roi) \
        .filter(ee.Filter.date(start_date, end_date)) \
        .first()
    
    # Isolate the Land Cover layer
    landCover = dataset.select('Land_Cover')
    
    # If Land Cover image is found, clip it to the roi
    if landCover:
        clipped_image = landCover.clip(roi)

    legend = {
        'Trees': '#005e00',
        'Grass': '#ffff00',
    }

    # Add the clipped Land Cover image and land cover legend
    Map.add_layer(clipped_image, {}, 'Land Cover', opacity=0.35)
    Map._add_legend(title='Land Type', legend_dict=legend, position='bottomright')

    # Return the map
    return Map


def return_neonrgb_data (map, start_date, end_date, roi):
    """
    Returns NEON RGB data for a given site and date range

    Args:
        map: a geemap Map object
        start_date: YYYY-MM-DD formatted string
        end_date: YYYY-MM-DD formatted string
        roi: region of interest in the form of an ee.Geometry.BBox object

    Returns:
        Geemap layer object of NEON RGB data for a given site
    """
    Map = map
    
    ee.Initialize(project='agedgeeffects')

    # NEON sites here: https://www.neonscience.org/field-sites/explore-field-sites
    dataset = ee.ImageCollection("projects/neon-prod-earthengine/assets/RGB/001") \
        .filterDate(start_date, end_date) \
        .filter('NEON_SITE == "UKFS"').mosaic()
    

    Map.addLayer(ee.FeatureCollection([ee.Feature(roi)]), {}, 'ROI', False)   # ROI checking layer

    return Map