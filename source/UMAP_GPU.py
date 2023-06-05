#from osgeo import gdal
import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
#import umap
import cuml
#import umapp

# output_dims = 20

# mat = loadmat("data/Danubio/PRS_L2D_STD_20220725095308_20220725095312_0001_VNIR_Cube.mat")
# HS_image = np.array(mat["PRISMA_data"], dtype=np.float32)
# HS_reshaped = np.reshape(HS_image, (-1,HS_image.shape[0]))

output_dims = 8

mat = loadmat("data/California/UiT_HCD_California_2017.mat")
HS_image = np.array(mat["t1_L8_clipped"], dtype=np.float32)
HS_reshaped = np.reshape(HS_image, (-1,HS_image.shape[2]))
reducer = cuml.UMAP(n_components= output_dims)

print("commincia UMAP")
embedding = reducer.fit_transform(HS_reshaped)
print("finisce UMAP")

# HS_embedded = np.reshape(embedding,(HS_image.shape[1],HS_image.shape[2],output_dims))
HS_embedded = np.reshape(embedding,(HS_image.shape[0],HS_image.shape[1],output_dims))

#normalization for visualization
HS_embedded_scaled = HS_embedded
for i in range(output_dims):
    HS_embedded_scaled[:,:,i] = HS_embedded[:,:,i]/HS_embedded[:,:,i].max()

# plt.imshow(HS_embedded_scaled[:,:,0:3])
# plt.show()

# mat["PRISMA_data"] = HS_embedded#_scaled
# savemat("data/California/PRS_L2D_STD_20220725095308_20220725095312_0001_VNIR_Cube_20ch.mat",mat)
mat["t1_L8_clipped"] = HS_embedded#_scaled
savemat("data/California/UiT_HCD_California_2017_reduced_6ch_linux.mat",mat)