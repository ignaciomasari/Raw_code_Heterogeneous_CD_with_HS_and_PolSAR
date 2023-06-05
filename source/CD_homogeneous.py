import numpy as np
import scipy
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
from PIL import Image

def filtering(d, t1, t2):
    # print("Filtering!")
    d = d[..., np.newaxis]
    d = np.concatenate((d, 1.0 - d), axis=2)
    W = np.size(d, 0)
    H = np.size(d, 1)
    stack = np.concatenate((t1, t2), axis=2)
    CD = dcrf.DenseCRF2D(W, H, 2)
    d[d == 0] = 10e-20
    U = -(np.log(d))
    U = U.transpose(2, 0, 1).reshape((2, -1))
    U = U.copy(order="C")
    CD.setUnaryEnergy(U.astype(np.float32))
    pairwise_energy_gaussian = create_pairwise_gaussian((10, 10), (W, H))
    CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=1)
    pairwise_energy_bilateral = create_pairwise_bilateral(
        sdims=(10, 10), schan=(0.1,), img=stack, chdim=2
    )
    CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=1)
    Q = CD.inference(3)
    heatmap = np.array(Q)
    heatmap = np.reshape(heatmap[0, ...], (W, H))
    return heatmap

if __name__=='__main__':
    CSK_before_path = './data/Danubio/CSKS2_GTC_B_HI_0B_HH_RD_SF_20200908_clipped_resamp_pp.mat'
    CSK_after_path = './data/Danubio/CSKS1_20210903_clipped_resampled_pp.mat'

    mat_before = scipy.io.loadmat(CSK_before_path)
    t1 = np.array(mat_before["CSK"], dtype=float)
    t1 = np.expand_dims(t1.squeeze(),-1)

    mat_after = scipy.io.loadmat(CSK_after_path)
    t2 = np.array(mat_after["CSK"], dtype=float)
    t2 = np.expand_dims(t2.squeeze(),-1)

    d = t1 - t2
    d = np.linalg.norm(d, 2, -1)

    d[d > np.mean(d) + 3.0 * np.std(d)] = np.mean(d) + 3.0 * np.std(d)
    d = d / np.max(d)

    heatmap = filtering(d, t1, t2)
    otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))
    # otsu = 
    CD_map = heatmap >= otsu  # CD_map = heatmap >= local_otsu
    CD_map_int = np.array(CD_map * 255, dtype=np.uint8)

    CD_path = './data/Danubio/HomogeneousCD.png'
    img = Image.fromarray(CD_map_int)
    img.save(CD_path)

    plt.imshow(CD_map)
    plt.show()

    Synthetic_CSK_path = './data/Danubio/CSK_20200908_synthetic.mat'
    Synthetic_CSK = t2.copy()
    Synthetic_CSK[CD_map==1] = t1[CD_map==1]
    plt.imshow(Synthetic_CSK)
    plt.show()

    mat_sar = {'CSK' : Synthetic_CSK}
    scipy.io.savemat(Synthetic_CSK_path , mat_sar)