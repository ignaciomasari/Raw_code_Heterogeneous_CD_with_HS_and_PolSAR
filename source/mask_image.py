# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio as rio


# %%
parent_folder = "C:/Users/Ignacio Masari"
# parent_folder = "C:/Users/lucam/IM"
mask_path = parent_folder + "/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/CSG/mask.png"
mask = cv2.imread(mask_path)
# mask = ((mask / 255) - 1) * (-1)
mask = mask / 255
# plt.imshow(mask)
# plt.show()


# %%
# CSG_path = parent_folder + "/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/CSG/CSG_SSAR2_GTC_B_0404_STR_007_HH_RD_F_20231004170859_20231004170905_Clipped.tif"
CSG_path = parent_folder + "/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/CSG/CSG_SSAR2_GTC_B_0404_STR_007_HV_RD_F_20231004170859_20231004170905_Clipped.tif"

with rio.open(CSG_path) as src:
    image_original = src.read().squeeze()
    profile = src.profile

# print(profile)

# plt.imshow(image_original)
# plt.show()

# %% modify image
image_mod = image_original.copy()

# 1st modification
red_pixel_coords = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0))
blue_pixel_coords = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
image_mod[blue_pixel_coords] = image_mod[red_pixel_coords]

# 2nd modification
yellow_pixel_coords = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
magenta_pixel_coords = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
image_mod[yellow_pixel_coords] = image_mod[magenta_pixel_coords]

# 3rd modification
green_pixel_coords = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
center = int(np.mean(green_pixel_coords[0])), int(np.mean(green_pixel_coords[1]))
center = np.array(center, dtype=np.float32)

for x, y in zip(green_pixel_coords[0], green_pixel_coords[1]):
    pos = np.array((x,y), dtype=np.int32)
    direction = center - pos
    direction = direction / np.linalg.norm(direction)
    sample_pos = pos + np.array(direction * 30, dtype=np.int32)
    image_mod[pos[0], pos[1]] = image_mod[sample_pos[0], sample_pos[1]]

# image_dif = image_original - image_mod
# image_dif[image_dif!=0] = 1
# image_mod[image_mod>0.6] = 0.6

# plt.imshow(image_mod / image_mod.max())
# plt.imshow(image_dif / image_dif.max())
# plt.show()
# %%
# CSG_path_mod = parent_folder + "/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/CSG/CSG_SSAR2_GTC_B_0404_STR_007_HH_RD_F_20231004170859_20231004170905_Clipped_mod.tif"
CSG_path_mod = parent_folder + "/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Bolsena/CSG/CSG_SSAR2_GTC_B_0404_STR_007_HV_RD_F_20231004170859_20231004170905_Clipped_mod.tif"

with rio.open(CSG_path_mod, 'w', **profile) as dst:
    dst.write(image_mod, 1)
    # print(dst.crs)
