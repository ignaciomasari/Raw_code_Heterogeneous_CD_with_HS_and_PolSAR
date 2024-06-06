import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio


CSG_DualPol_path_mod = "data/Bolsena_30m/CSG_SSAR2_GTC_B_DualPol_RD_F_20231004_Clipped_mod_resamp.tif"

with rio.open(CSG_DualPol_path_mod) as src:
    image_mod = src.read().squeeze()
    image_mod = np.moveaxis(image_mod, 0, -1)
    profile_mod = src.profile
    image_hh_mod = image_mod[:,:,0].squeeze()
    print(image_hh_mod.shape)


CSG_HH_path_original = "data/Bolsena_30m/CSG_SSAR2_GTC_B_0404_STR_007_HH_RD_F_20231004170859_20231004170905_Clipped_resamp.tif"

with rio.open(CSG_HH_path_original) as src:
    image_hh_original = src.read().squeeze()
    profile_hh_original = src.profile
    print(image_hh_original.shape)

difference = np.abs(image_hh_original - image_hh_mod)
difference = difference != 0
difference = difference.astype(np.uint8)
# plt.imshow(difference)
# plt.show()

# mark as change also the borders bc of registration of PRISMA images
difference[:4,:] = 1
difference[:,-3:] = 1


# Save the difference image as a TIFF file
output_path = "data/Bolsena_30m/GT_difference.tif"
with rio.open(output_path, 'w', **profile_hh_original) as dst:
    dst.write(difference, 1)
