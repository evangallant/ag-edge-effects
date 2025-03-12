import ee

# Initialize Earth Engine
ee.Initialize(project='agedgeeffects')

# Define our region of interest (ROI) for the project
roi = ee.geometry.Rectangle(-95.245479, 39.018359, -95.144882, 39.081559)