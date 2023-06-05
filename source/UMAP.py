
# If using intel herdware, use optimization
try:
    from sklearnex import patch_sklearn
except ImportError:
    print("Not Intel-optimized")
else:
    patch_sklearn()
    print("Intel-optimized")

import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from utils import postprocess
#import umap
import umapp


def reduce(output_dims):
    mat = loadmat("data/Danubio/PRS_L2D_STD_20210909_VNIR_Cube_clipped_registered_pp.mat")
    VNIR_image = np.array(mat["VNIR"], dtype=np.float32)
    mat = loadmat("data/Danubio/PRS_L2D_STD_20210909_SWIR_Cube_clipped_registered_pp.mat")
    SWIR_image = np.array(mat["SWIR"], dtype=np.float32)
    #in the new dataset, image is b,w,h. In the old one is w,h,b
    HS_image = np.concatenate((VNIR_image, SWIR_image), axis=0)
    HS_image = np.moveaxis(HS_image, source=0, destination=-1)
    HS_flat = np.reshape(HS_image, (-1,HS_image.shape[2]))

    reducer = umapp.UMAP(n_components=output_dims)
    print("commincia UMAP")
    embedding = reducer.fit_transform(HS_flat)
    print("finisce UMAP")

    # HS_embedded = np.reshape(embedding,(HS_image.shape[1],HS_image.shape[2],output_dims))
    HS_embedded = np.reshape(embedding,  (HS_image.shape[0], HS_image.shape[1], output_dims))

    mat["PRISMA"] = HS_embedded
    savemat(f"data/Danubio/PRISMA_{output_dims}ch.mat",mat)
    postprocess(f"data/Danubio/PRISMA_{output_dims}ch", "PRISMA")

    #normalization for visualization
    # HS_embedded_scaled = HS_embedded
    # for i in range(output_dims):
    #     HS_embedded_scaled[:,:,i] = (HS_embedded[:,:,i] - HS_embedded[:,:,i].min())/(HS_embedded[:,:,i].max() - HS_embedded[:,:,i].min())

    # plt.imshow(HS_embedded_scaled[:,:,0:3])
    # plt.show()
    # del mat

if __name__ == "__main__":
    # reduce(1)
    # reduce(3)
    reduce(5)
    reduce(10)
    reduce(12)
    reduce(15)
    reduce(20)
    reduce(25)
    reduce(30)
    reduce(40)
    reduce(50)