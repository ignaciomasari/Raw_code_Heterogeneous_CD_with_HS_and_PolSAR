
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# from tensorflow.python.util import deprecation

# deprecation._PRINT_DEPRECATION_WARNINGS = False

import scipy.io
import os
import sys
import time
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sklearn.metrics as mt
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mat = scipy.io.loadmat("data/Danubio/CSKS1_GTC_B_HI_0B_HH_RD_SF_20210701170109_20210701170116_filt_clipped_stacked_pp.mat")
t1 = np.array(mat["CSK"], dtype=float)
t1 = np.expand_dims(t1.squeeze(),-1)
mat2 = scipy.io.loadmat("data/Danubio/PRISMA_6ch.mat")
t2 = np.array(mat2["PRISMA"], dtype=float)

# plt.imshow(t1)
# plt.show()

# plt.imshow(t2)
# plt.show()

plt.hist(x=t1.ravel(), bins='auto',
        alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('CSK')
plt.show()

fig, axes = plt.subplots(t2.shape[2])
fig.suptitle("PRISMA")

if (t2.shape[2] == 1):
    axes.hist(x=t2.ravel(), bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    axes.grid(axis='y', alpha=0.75)
else:
    for i in range(t2.shape[2]):
        axes[i].hist(x=t2[:,:,i].ravel(), bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        axes[i].grid(axis='y', alpha=0.75)
        # axes[i].xlabel('Value')
        # axes[i].ylabel('Frequency')
        # axes[i].title('Histogram')

plt.show()

mat = loadmat("data/Danubio/PRS_L2D_STD_20220725095308_20220725095312_0001_VNIR_Cube_clip_stacked_registered.mat")
VNIR_image = np.array(mat["VNIR"], dtype=np.float32)
mat = loadmat("data/Danubio/PRS_L2D_STD_20220725095308_20220725095312_0001_SWIR_Cube_clip_stacked_registered.mat")
SWIR_image = np.array(mat["SWIR"], dtype=np.float32)
#in the new dataset, image is b,w,h. In the old one is w,h,b
HS_image = np.concatenate((VNIR_image, SWIR_image), axis=0)
HS_image = np.moveaxis(HS_image, source=0, destination=-1)
HS_flat = np.reshape(HS_image, (-1,HS_image.shape[2]))


fig, ax = plt.subplots()
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.75)
for i in range(HS_flat.shape[1]):
    ax.clear()

    heights,edges = np.histogram(HS_flat[:,i], bins=100)
    edges = edges[:-1]+(edges[1]-edges[0]) 

    ax.plot(edges,heights) 
    ax.set_title(f'PRISMA band {i}')
    # plt.ylim((0,9000))
    plt.pause(0.1)
    time.sleep(0.05)
