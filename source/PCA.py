
# If using intel herdware, use optimization
try:
    from sklearnex import patch_sklearn
except ImportError:
    print("Not Intel-optimized")
else:
    patch_sklearn()
    print("Intel-optimized")

import numpy as np
import os
#import matplotlib.pyplot as plt
import sklearn.decomposition
from skimage.measure import block_reduce


def reduce_kPCA(HS_image, output_dims, kernel='linear', spatial_reduction=6):
   
    HS_image_reduced = block_reduce(HS_image, block_size=(spatial_reduction, spatial_reduction, 1), func=np.mean)

    HS_flat_reduced = np.reshape(HS_image_reduced, (-1,HS_image_reduced.shape[2]))

    HS_flat = np.reshape(HS_image, (-1,HS_image.shape[2]))

    reducer = sklearn.decomposition.KernelPCA(n_components=output_dims, kernel=kernel)
    print("commincia PCA")
    reducer.fit(HS_flat_reduced)
    print("finisce training PCA")
    embedding = reducer.transform(HS_flat)
    print("finisce PCA")

    # HS_embedded = np.reshape(embedding,(HS_image.shape[1],HS_image.shape[2],output_dims))
    HS_embedded = np.reshape(embedding,  (HS_image.shape[0], HS_image.shape[1], output_dims))

    return HS_embedded


if __name__ == "__main__":
    
    DATASET = "E_R2"

    for file in os.listdir(f"data/{DATASET}"):
        if file.endswith(".npy"):
            if 'VNIR' in file:
                VNIR_path = f"data/{DATASET}/{file}"
            elif 'SWIR' in file:
                SWIR_path = f"data/{DATASET}/{file}"            

    VNIR_image = np.array(np.load(VNIR_path), dtype=np.float32)
    SWIR_image = np.array(np.load(SWIR_path), dtype=np.float32)
    
    HS_image = np.concatenate((VNIR_image, SWIR_image), axis=0)
    #in the new dataset, image is b,w,h. In the old one is w,h,b
    HS_image = np.moveaxis(HS_image, source=0, destination=-1)
    # HS_image = np.concatenate((VNIR_image, SWIR_image), axis=-1)
    # HS_image = np.moveaxis(HS_image, source=0, destination=-1)
    
    if DATASET == "E_R2":
        HS_image = HS_image[1:,:-4,:] #remove the first row  and the last 4 columns (no_data from the registration)

    output_dims = [6, 7, 9]
    kernel = 'linear'

    HS_embedded = reduce_kPCA(HS_image, np.max(np.array(output_dims)), kernel=kernel)

    # import matplotlib.pyplot as plt
    # plt.imshow(HS_embedded[:,:,0:3])
    # plt.show()

    for dim in output_dims:
        np.save(f"data/{DATASET}/PRISMA_kPCA_{kernel}_{dim}ch.npy",HS_embedded[:,:,:dim])
    