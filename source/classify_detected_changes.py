import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from PIL import Image
import pandas as pd
from skimage.filters import threshold_otsu

GT_before_path = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Data/coordinates_Danubio/gt_cd/gt20200411_v2_clipped.tif"
with rio.open(GT_before_path) as src:
    GT_before = src.read().squeeze()
GT_before[GT_before==255]=0

GT_after_path = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Data/coordinates_Danubio/gt_cd/gt20210909_clipped.tif"
with rio.open(GT_after_path) as src:
    GT_after = src.read().squeeze()
GT_after[GT_after==255]=0

labels=np.unique(GT_before)
if np.not_equal(labels,np.unique(GT_after)).any():
    raise Exception("not same clases in both GT")

labels = labels[1:]

confusion_map_png_path = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Code/ChangeDetection/repo2/Deep_image_translation_modified/Results/X-Net/Danubio_10ch/Confusion_map.png"
confusion_map_image = Image.open(confusion_map_png_path)
confusion_map = np.array(confusion_map_image)[:,:,1]

confusion_change = np.zeros((len(labels), len(labels)))
confusion_no_change = np.zeros((len(labels), len(labels)))

for ib, label_before in enumerate(labels):
    for ia, label_afeter in enumerate(labels):
        mask = np.bitwise_and(GT_before==label_before, GT_after==label_afeter)
        if mask.sum()==0:
            continue
        confusion_change[ib,ia] = np.bitwise_and(confusion_map, mask).sum()
        confusion_no_change[ib,ia] = np.bitwise_and(~confusion_map, mask).sum()

print(labels)

# Create a pandas DataFrame for each matrix
df1 = pd.DataFrame(confusion_change)
df2 = pd.DataFrame(confusion_no_change)

# Create a Pandas Excel writer using the openpyxl engine
writer = pd.ExcelWriter('C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Code/ChangeDetection/repo2/Deep_image_translation_modified/Results/change_matrices.xlsx', engine='xlsxwriter')

# Write each DataFrame to a separate sheet in the Excel file
df1.to_excel(writer, sheet_name='confusion_change', header=labels)
df2.to_excel(writer, sheet_name='confusion_no_change', header=labels)

# Save the Excel file
writer.save()



