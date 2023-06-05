import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio as rio
from skimage.filters import threshold_otsu

heatmap_path = "./Results_masked_10pr_521e/X-Net/Danubio/d_filtered.png" 
heatmap = np.array(Image.open(heatmap_path))[:,:,1]
otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))
CD_map = heatmap >= otsu  # CD_map = heatmap >= local_otsu

plt.imshow(CD_map)
plt.show()

heatmap_path = "./Results_masked_10pr_521e/X-Net/Danubio/d_y.png" 
heatmap = np.array(Image.open(heatmap_path))[:,:,1]
otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))
CD_map = heatmap >= otsu  # CD_map = heatmap >= local_otsu

plt.imshow(CD_map)
plt.show()