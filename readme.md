# **Upscaling the USFS Land Cover Classification Dataset in Colorado**
## **Background:**
The USFS annually produces a map of land cover classes and landscape change across the coterminus United States. From the dataset description:
"""
Outputs include three annual products: change, land cover, and land use. The change model output relates specifically to vegetation cover and includes slow loss, fast loss (which also includes hydrologic changes such as inundation or desiccation), and gain. These values are predicted for each year of the Landsat time series and serve as the foundational products for LCMS. We apply a ruleset based on ancillary datasets to create the final change product, which is a refinement/reclassification of the modeled change to 15 classes that explicitly provide information on the cause of landscape change (e.g., Tree Removal, Wildfire, Wind damage). Land cover and land use maps depict life-form level land cover and broad-level land use for each year.
<br>
Because no algorithm performs best in all situations, LCMS uses an ensemble of models as predictors, which improves map accuracy across a range of ecosystems and change processes (Healey et al., 2018). The resulting suite of LCMS change, land cover, and land use maps offer a holistic depiction of landscape change across the United States since 1985.
"""

## **Motivation:**
The resulting map has a 30x30m resolution, and while generally accurate, produces clearly inaccurate results at small scales. Shown here is the USFS dataset overlain on high resolution satellite imagery of an area near Leadville, CO:
<br>
![alt text](/figs/tree_misidentifications.png)
<br>
Clearly the USFS data identifies a significant portion of open grassy shrubland on the left side of the image as trees (darker blue). This should likely be classified as grass/forb/herb & shrubs mix. Similar small scale inaccuracies can be seen virtually everywhere, especially on the boundaries between different land classes, as well as with small scale features like creeks or clearings.
<br>
The USFS product is extremely helpful at quantifying general land cover trends at large scales, but could be improved upon at smaller scales to provide better context to decision makers and organizations operating at a similarly local level. The original dataset, despite consistently exhibiting these inaccuracies, contains enough accurate information to allow computer vision models to learn the relationships from the correct predictions while not overfitting to the small scale mistakes.

## **Approach:**
In order to create an upscaled land classification map the data used by the model needs to be higher resolution as well. For this task we use 10x10m resolution Sentinel 2 satellite imagery (https://developers.google.com/earth-engine/datasets/catalog/sentinel-2).
This dataset provides red, green, blue, and NIR bands at 10m resolution (as well as many other bands which were not used to train the model), and is ideal for providing the model context about vegetative patterns on a small scale.
Training data included 150,000 samples from 20 different ~10x10 mile ROIs in Colorado, and as such is best suited for application to Colorado and similar surrounding regions in the Rocky Mountains and Great Plains.
<br>
Model architecture development was informed by the hypothesis that information about the area surrounding a given target pixel is critical to identifying the land use class at that pixel. For example, take a single bright green pixel - in isolation it could be attributed to a lush corn field, a tree, a football field, etc. Only by looking at the surrounding area can you make sense of what is happing at the center of the image.
<br>
In this project, providing regional context is achieved by training a convolutional neural net (CNN) on 80x80 pixel images (pixel blocks) of the target pixel and its surrounding area. The model takes as input the red, green, blue, and NIR bands from the Sentinel 2 imagery, as well as a calculated NDVI band of the pixel block (resulting in a 5x80x80 ChannelsxHeightxWidth tensor), and returns a discrete classification for the central pixel. 
The CNN is extremely basic and contains 3 convolutional layers, each of which aim to identify relevant features at different scales within the pixel block. While training, the model compares the output at the 10x10m resolution pixel with the USFS class at the same location. Despite the small scale innaccuracies of the USFS data described above, this approach yields a model that learns from contiguous swathes of correctly labeled classes and (for the most part) ignores inaccurate edge cases, as they constitute an insignificant amount of the training data.

## **Findings:**
Below are examples of the model outputs on ROIs in the same region as the training data where USFS data exists, and ROIs in various different parts of the world without USFS data to compare to.

### **Against ROIs with USFS data to compare to:**
Here we look at a number of ROIs in Colorado which have similar Rocky Mountain-esque terrain as was prevelant in the training data.

**Snowmass:**

| ROI | Raw Sentinel 2 Imagery | Overlain Predictions | Predictions | USFS Data | Notes |
| Snowmass | ![alt text](/figs/snow_s2.png) | ![alt text](/figs/snow_5050.png) | ![alt text](/figs/snow_preds.png) | ![alt text](/figs/snow_usfs.png) | Despite the imperfect nature of the USFS data, the capacity of this model is shown here. The model accurately recognizes and categorizes the differences in vegetation types between north and south facing slopes in foothill features, and not only provides an upscaled map of the USFS data, but also addresses the small scale imperfections that define the low resolution approach. Here we see the model carefully discern between sparesly vegetated southern hill slopes and the densely forested northern sides while sticking the the same general categorization scheme as is present in the USFS data. |
| Boise | ![alt text](/figs/idaho_s2.png) | ![alt text](/figs/idaho_5050.png) | ![alt text](/figs/idaho_preds.png) | ![alt text](/figs/idaho_usfs.png) | However, it's not perfect. Again the model is very aware of the landscape and vegetation density, but sometimes makes radically different overall categorizations than the USFS data. This is likely due to differences in when the images used to make predictions were taken (as the USFS data contains many 'Snow' predictions), as well as the fact that this ROI is in Idaho, as opposed to Colorado. |

### **What it thinks of an ROI in a completely different area:**
And of course, when applied to diffent parts of the world the model breaks down. It still recognizes general features, but doesn't characterize them accurately.

**Southern Canada:**
Here the model recognizes that some areas of the image are highly photosynthetically productive, but attributes it to a mixed composition of trees and shrubs, instead of just highly productive marshland surrounding a lake.

Raw Sentinel 2 imagery:
<br>
![alt text](/figs/canada_s2.png)
<br>
Overlain predictions:
<br>
![alt text](/figs/canada_5050.png)
<br>
Predictions:
<br>
![alt text](/figs/canada_preds.png)
<br>

**Sahara:**
When provided with an image of the Sahara Desert, it predicts that everything is either Trees or Tall Shrubs and Trees Mix:

Raw Sentinel 2 imagery:
<br>
![alt text](/figs/sahara_s2.png)
<br>
Overlain predictions:
<br>
![alt text](/figs/sahara_5050.png)
<br>
Predictions:
<br>
![alt text](/figs/sahara_preds.png)
<br>

## **Conclusion:**
This approach demonstrates the benefits of providing surrounding context in the land use classification process. However, it is limited by:
- The localized and relatively small nature of the training data
- The non-seasonally agnostic approach to classification, which requires data from the summer
- The high computational cost and inefficiency of running a 5x80x80 tensor through a CNN in order to predict the class of a single pixel
- The need for a family of models to be trained in order to make accurate predictions in different ecological zones

And likely many other things. However, I consider this project a success, as I successfully applied a number of crucial geospatial data science skills and produced a (if limitedly) helpful model. Thank you for reading!