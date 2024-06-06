
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
#import umap
import umapp


def reduce(HS_image, output_dims):
    
    # HS_image = HS_image[1:,:-4,:] #remove the first row  and the last 4 columns (no_data from the registration)
    HS_flat = np.reshape(HS_image, (-1,HS_image.shape[2]))

    reducer = umapp.UMAP(n_components=output_dims)
    print("commincia UMAP")
    embedding = reducer.fit_transform(HS_flat)
    print("finisce UMAP")

    # HS_embedded = np.reshape(embedding,(HS_image.shape[1],HS_image.shape[2],output_dims))
    HS_embedded = np.reshape(embedding,  (HS_image.shape[0], HS_image.shape[1], output_dims))

    #normalization for visualization
    # HS_embedded_scaled = HS_embedded
    # for i in range(output_dims):
    #     HS_embedded_scaled[:,:,i] = (HS_embedded[:,:,i] - HS_embedded[:,:,i].min())/(HS_embedded[:,:,i].max() - HS_embedded[:,:,i].min())

    # plt.imshow(HS_embedded_scaled[:,:,0:3])
    # plt.show()
    # del mat

    return HS_embedded

if __name__ == "__main__":
    # VNIR_path = "data/Bolsena/PRS_L2D_STD_20231010101445_20231010101449_0001_VNIR_Cube_Clipped_registered.npy"
    # SWIR_path = "data/Bolsena/PRS_L2D_STD_20231010101445_20231010101449_0001_SWIR_Cube_Clipped_registered.npy"

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

    if DATASET == "E_R2":
        HS_image = HS_image[1:,:-4,:] #remove the first row  and the last 4 columns (no_data from the registration)

    # output_dims = [1, 2, 3, 5, 8, 10]#, 12, 15, 20, 25, 30, 40]
    output_dims = [6,7,9]

    for reduced_dim in output_dims:
        np.save(f"data/{DATASET}/PRISMA_UMAP_{reduced_dim}ch.npy", reduce(HS_image, reduced_dim))
        # postprocess(f"data/E_R/PRISMA_{output_dims}ch")
    