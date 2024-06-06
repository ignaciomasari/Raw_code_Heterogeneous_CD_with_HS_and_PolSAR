import numpy as np

from scipy.io import loadmat
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def print_mat(path, name, ch0=0):
    mat = loadmat(path)
    img = mat[name]
    # PRISMA_img = np.moveaxis(PRISMA_img,source= 0, destination= -1)
    img = (img + 1) / 2

    plt.imshow(img[:,:,ch0:ch0+3])
    plt.show()

def postprocess(filename, log = False):
    image = np.load(filename + ".npy")

    #apply log
    if (log):
        image = np.log(image + 0.1)
    #calc stats
    mean = image.mean()
    sd = image.std()
    #cut off outliers
    image[image > mean + 3*sd] = mean + 3*sd
    image[image < mean - 3*sd] = mean - 3*sd
    maxim = image.max()
    minim = image.min()
    center = (minim + maxim) / 2
    #normalize between -1 and 1
    image = (image - center) / (maxim - center)

    np.save(filename + "_pp.npy", image)

def tif2npy_folder(folder_path, all_bands_save=False):

    import rasterio as rio
    
    images_path = os.listdir(folder_path)
    sar_image_path = (folder_path + ([x for x in images_path if "CSK" in x or "CSG" in x])[0])[:-4]

    sar_file = rio.open(sar_image_path+".tif")
    sar_data = sar_file.read()
    sar_file.close()
    sar_data = np.moveaxis(sar_data,0,-1)
    sar_data = sar_data[1:,:-4,:]
    np.save(sar_image_path + ".npy", sar_data)
    postprocess(sar_image_path, False)#or should be true??


    VNIR_image_path = (folder_path + ([x for x in images_path if "VNIR" in x])[0])[:-4]
    SWIR_image_path = (folder_path + ([x for x in images_path if "SWIR" in x])[0])[:-4]

    VNIR_data = rio.open(VNIR_image_path + ".tif") 
    VNIR_image = VNIR_data.read()
    VNIR_data.close()
    VNIR_data = np.moveaxis(VNIR_image,0,-1)
    VNIR_image = VNIR_data[1:,:-4,:]
    np.save(VNIR_image_path + ".npy", VNIR_image)
    postprocess(VNIR_image_path)

    SWIR_data = rio.open(SWIR_image_path + ".tif") 
    SWIR_image = SWIR_data.read()
    SWIR_data.close()
    SWIR_data = np.moveaxis(SWIR_image,0,-1)
    SWIR_image = SWIR_data[1:,:-4,:]
    np.save(SWIR_image_path + ".npy", SWIR_image)  
    postprocess(SWIR_image_path)

    if (all_bands_save):
        all_band_raster = np.concatenate((VNIR_image,SWIR_image), axis=-1)
        # all_band_image = np.reshape(all_band_image,(all_band_image.shape[1],all_band_image.shape[2],all_band_image.shape[0])) #reshape as image
        # all_band_image = np.moveaxis(all_band_raster,0,-1)
        np.save(VNIR_image_path.replace("VNIR","Whole") + ".npy", all_band_raster)
        postprocess(VNIR_image_path.replace("VNIR","Whole"))

def tif2npy_file(file_path, log=False):

    img_file = rio.open(file_path)
    img_data = img_file.read()
    img_file.close()
    img_data = np.moveaxis(img_data,0,-1)
    # only for E_R2
    # img_data = img_data[1:,:-4,:]
    if (file_path.endswith(".tif")):
        file_name = file_path[:-4]
    elif (file_path.endswith(".tiff")):
        file_name = file_path[:-5]

    np.save(file_name + ".npy", img_data)

    # postprocess(file_name, log)

def cut_data_window(input_path):
    import rasterio as rio
    from rasterio.windows import Window, get_data_window
    from rasterio.transform import transform

    if input_path.endswith(".tif"):
        output_path = input_path[:-4] + "_cut.tif"
    else:
        raise TypeError

    with rio.open(input_path) as src:
        profile = src.profile.copy()
        image = src.read()
        i = image[0]

        data_window = get_data_window(src.read(masked=True), nodata=0)
        data_transform = transform(data_window, src.transform)
        profile.update(
            transform=data_transform,
            height=data_window.height,
            width=data_window.width)

        data = src.read(window=data_window)

    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

def correct_profile(input_path, profile_path, output_path):
    import rasterio as rio

    with rio.open(input_path, 'r') as src:          
        image = src.read()
    with rio.open(profile_path, 'r') as src:          
        profile = src.profile
    with rio.open(output_path, 'w', **profile) as dst:          
        dst.write(image)

def georeference_png_file(profile_sample, results_path, correct_image=False):
    import rasterio as rio
    # profile_sample might be a profile or the path to an image with a profile to sample (NOT TESTED)
    if isinstance(profile_sample, str):
        with rio.open(profile_sample, 'r') as src:          
            profile = src.profile
    else:
        profile = profile_sample
        
    confusion_image = Image.open(results_path,mode='r')

    np_results = np.reshape(np.array(confusion_image.getdata(band=1)),(1, confusion_image.height, confusion_image.width)) / 255

    if correct_image:
        z = np.zeros((1,832,4))
        np_results = np.concatenate((np_results, z),axis=2)
        z = np.zeros((1,1,762))
        np_results = np.concatenate((z, np_results),axis=1)
        profile["count"] = 1

    with rio.open(results_path[:-4] + "_geo.tif", 'w', **profile) as dst:          
        dst.write(np_results)

def georeference_npy_file(profile_sample, results_path, correct_image=False):
    import rasterio as rio
    # profile_sample might be a profile or the path to an image with a profile to sample (NOT TESTED)
    if isinstance(profile_sample, str):
        with rio.open(profile_sample, 'r') as src:          
            profile = src.profile
    else:
        profile = profile_sample
        
    np_results = np.load(results_path)

    if np_results.ndim == 4:
        np_results = np_results[0,...]
    elif np_results.ndim == 2:
        np_results = np_results[np.newaxis,...]

    if correct_image:
        z = np.zeros((832,4, np_results.shape[2]))
        np_results = np.concatenate((np_results, z),axis=1)
        z = np.zeros((1,762, np_results.shape[2]))
        np_results = np.concatenate((z, np_results),axis=0)
        
    profile["count"] = np_results.shape[2]

    with rio.open(results_path[:-4] + "_geo.tif", 'w', **profile) as dst:   
            dst.write(np.moveaxis(np_results,-1,0))

def confusion_matrix(ground_truth_path, predicted_path):
    import rasterio as rio

    with rio.open(ground_truth_path, 'r') as src:          
        gt = (np.array(src.read())).squeeze()

    with rio.open(predicted_path, 'r') as src:          
        change_map = (np.array(src.read())).squeeze()

    #WHY NOT WORKING?
    # true_negative = np.array((change_map == 0) == (gt == 1)).sum()
    # false_negative = np.array((change_map == 0) == (gt == 2)).sum()
    # true_positive = np.array((change_map == 1) == (gt == 2)).sum()
    # false_positive = np.array((change_map == 1) == (gt == 1)).sum()

    true_negative2 = 0
    false_negative2 = 0
    true_positive2 = 0
    false_positive2 = 0
    no_data = 0

    for i in range(change_map.shape[0]):
        for j in range(change_map.shape[1]):
            if (change_map[i,j] == 0) and (gt[i,j]==1):
                true_negative2 += 1
            elif (change_map[i,j] == 0) and (gt[i,j]==2):
                false_negative2 += 1
            elif (change_map[i,j] == 1) and (gt[i,j]==2):
                true_positive2 += 1
            elif (change_map[i,j] == 1) and (gt[i,j]==1):
                false_positive2 += 1
            else:
                no_data += 1

    data = change_map.size - no_data
    sensitivity = true_positive2 / data
    specificity = false_positive2 / data
    results = {
        "TN": true_negative2,
        "FN": false_negative2,
        "FP": false_positive2,
        "TP": true_positive2,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
    return results

def filtering(d, t1, t2):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral
    from pydensecrf.utils import create_pairwise_gaussian
    
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

def compare_homogeneous_images(x, y):
    from skimage.filters import threshold_otsu
    
    d = x - y
    d = np.linalg.norm(d, 2, -1)    

    d[d > np.mean(d) + 3.0 * np.std(d)] = np.mean(d) + 3.0 * np.std(d)
    d = d / np.max(d)

    heatmap = filtering(d, x, y)
    otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))        
    CD_map = heatmap >= otsu
    return CD_map

if __name__ == "__main__":
    
    # folder_name = "././data/E_R/"
    # tif2npy_folder(folder_name, True)
    # pre_sentinel = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Emilia_Romagna/Sentinel/S2A_MSIL2A_20220717/GRANULE/L2A_T32TQQ_A036913_20220717T100925/IMG_DATA/R20m/T32TQQ_20220717T100611_resampled_clip.tif"
    # tif2npy_file(pre_sentinel)
    # post_sentinel = "C:/Users/lucam/IM/OneDrive - unige.it/UGenoa/Projects/PRISMALearn/Immagini/Emilia_Romagna/Sentinel/S2A_MSIL2A_20230523/GRANULE/L2A_T32TQQ_A041346_20230523T100828/IMG_DATA/R20m/T32TQQ_20230523T100601_resampled_clip.tif"
    # tif2npy_file(post_sentinel)
    # pre_path = "data/E_R/T32TQQ_20220717T100611_resampled_clip_pp.npy"
    # post_path = "data/E_R/T32TQQ_20230523T100601_resampled_clip_pp.npy"
    # pre = np.load(pre_path)
    # post = np.load(post_path)
    # CD_map = compare_homogeneous_images(pre, post)
    # plt.imshow(CD_map)
    # plt.show()

    # np.save("data/E_R/Sentinel_CD_GT_pp.npy", CD_map)
    # gt_path_sentinel = "data/E_R/Sentinel_CD_GT_pp.npy"
    # overlap_path = "data/E_R/Overlap_het_homo.npy"
    # Change_map_path = "Results/X-Net/E_R_alpha_not_updated_3ch/Change_map.png"
    profile_path = "data\LUCCA\VNIR_clipped_registered_clip.tif"
    image_path = "data\LUCCA\PRISMA_kPCA_linear_2ch.npy"
    georeference_npy_file(profile_path, results_path=image_path, correct_image=False)

    # tif_file_path = "data/Bolsena_30m/CSG_SSAR2_GTC_B_DualPol_RD_F_20231004_Clipped_mod_resamp.tif"
    # tif2npy_file(tif_file_path)
    # tif_file_path = "data/Bolsena_30m/GT_difference.tif"
    # tif2npy_file(tif_file_path)

    
