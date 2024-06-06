import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

# with rio.open('./data/E_R/water_post_raster.tif') as src:    
#     water_post = src.read().squeeze()

# with rio.open('./data/E_R/water_pre_raster.tif') as src:    
#     water_pre = src.read().squeeze()

# with rio.open('./data/E_R/previous_observed_raster_GFM.tif') as src:    
#     GFM_GT = src.read().squeeze()

# with rio.open('./data/E_R/other_changer_het_raster.tif') as src:    
#     other_changes = src.read().squeeze()

# corrected_GT = np.zeros(water_post.shape, dtype=np.bool_)
# corrected_GT = np.logical_or(water_post, other_changes)
# corrected_GT = np.logical_or(corrected_GT, GFM_GT)
# corrected_GT = np.logical_and(corrected_GT, np.logical_not(water_pre))

# # f, axarr = plt.subplots(1,5) 
# # axarr[0].imshow(water_pre)
# # axarr[1].imshow(water_post)
# # axarr[2].imshow(other_changes)
# # axarr[3].imshow(GFM_GT)
# # axarr[4].imshow(corrected_GT)
# # plt.show()
# np.save('./data/E_R/corrected_GT.npy', corrected_GT[1:,:-4])

# corrected_GT_flood = np.zeros(water_post.shape, dtype=np.bool_)
# corrected_GT_flood = np.logical_or(water_post, GFM_GT)
# corrected_GT_flood = np.logical_and(corrected_GT_flood, np.logical_not(water_pre))
# np.save('./data/E_R/corrected_GT_flood.npy', corrected_GT_flood[1:,:-4])


with rio.open('./data/E_R/GT_corrected_311023.tif') as src:    
    all_changes_corrected_311023 = src.read().squeeze()

np.save('./data/E_R/GT_corrected_311023.npy', all_changes_corrected_311023[1:,:-4])