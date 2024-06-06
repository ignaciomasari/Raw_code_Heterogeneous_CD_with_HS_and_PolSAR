# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio as rio

# %%
# img_path = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/PRISMA/PRS_L2D_STD_20231010101445_20231010101449_0001/PRS_L2D_STD_20231010101445_20231010101449_0001_VNIR_Cube_Clipped_registered.tif"
img_path = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/PRISMA/PRS_L2D_STD_20231010101445_20231010101449_0001/PRS_L2D_STD_20231010101445_20231010101449_0001_SWIR_Cube_Clipped_registered.tif"

with rio.open(img_path) as src:
    image_original = src.read()
    profile = src.profile

# update the value crs of the profile
profile['crs'] = rio.crs.CRS.from_epsg(32632)

# %%
# img_path_mod = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/PRISMA/PRS_L2D_STD_20231010101445_20231010101449_0001/PRS_L2D_STD_20231010101445_20231010101449_0001_VNIR_Cube_Clipped_registered_proj.tif"
img_path_mod = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/PRISMA/PRS_L2D_STD_20231010101445_20231010101449_0001/PRS_L2D_STD_20231010101445_20231010101449_0001_SWIR_Cube_Clipped_registered_proj.tif"

with rio.open(img_path_mod, 'w', **profile) as dst:
    dst.write(image_original)
    print(dst.crs)


