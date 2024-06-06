# %%
import os
import numpy as np
import cv2
import rasterio as rio
import matplotlib.pyplot as plt


def tif_to_npy(path):
    with rio.open(path) as src:
        image = src.read().squeeze()
        # profile = src.profile

    np.save(path.replace(".tif", ".npy"), image)
    return

def tif_to_npy2(path1, path2):
    with rio.open(path1) as src:
        image = src.read().squeeze()
        # profile = src.profile
    
    image_2_bands = np.zeros((image.shape[0], image.shape[1], 2))
    image_2_bands[:,:,0] = image

    with rio.open(path2) as src:
        image = src.read().squeeze()
        # profile = src.profile
    
    image_2_bands[:,:,1] = image

    np.save(path1.replace(".tif", "2_bands.npy"), image_2_bands)
    return

# mask_path = "./data/Bolsena_30m/mask_resamp.bmp"
# mask = cv2.imread(mask_path)
# mask = mask / 255 # normalize
# mask = (mask - 1) * -1 # invert
# mask = (mask[:,:,2]).astype(np.uint8)
# # plt.imshow(mask)
# # plt.show()
# np.save(mask_path.replace(".bmp", ".npy"), mask)

# CSG_HH_path_mod = "./data/Bolsena_30m/CSG_SSAR2_GTC_B_0404_STR_007_HH_RD_F_20231004170859_20231004170905_Clipped_mod_resamp.tif"
# CSG_HV_path_mod = "./data/Bolsena_30m/CSG_SSAR2_GTC_B_0404_STR_007_HV_RD_F_20231004170859_20231004170905_Clipped_mod_resamp.tif"

# tif_to_npy(CSG_HH_path_mod)
# tif_to_npy(CSG_HV_path_mod)

PRISMA_VNIR_path = "data\LUCCA\VNIR_clipped_registered_clip.tif"
PRISMA_SWIR_path = "data\LUCCA\SWIR_clipped_registered_clip.tif"
tif_to_npy(PRISMA_VNIR_path)
tif_to_npy(PRISMA_SWIR_path)


# Sentinel1_VV = "data\LUCCA\S1_VV_clipped_clip.tif"
# Sentinel1_VH = "data\LUCCA\S1_VH_clipped_clip.tif"

# tif_to_npy2(Sentinel1_VV, Sentinel1_VH)

# PRISMA_VNIR_path = "./data/E_R2/PRS_L2D_STD_20220715102100_20220715102104_0001_VNIR_Cube_clip_registered.tif"
# PRISMA_SWIR_path = "./data/E_R2/PRS_L2D_STD_20220715102100_20220715102104_0001_SWIR_Cube_clip_registered.tif"

# tif_to_npy(PRISMA_VNIR_path)
# tif_to_npy(PRISMA_SWIR_path)




