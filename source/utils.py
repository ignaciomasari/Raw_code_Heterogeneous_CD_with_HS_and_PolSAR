import numpy as np
import rasterio as rio
from rasterio.windows import get_data_window, transform
from scipy.io import loadmat, savemat
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

def postprocess(filename, matname, log = False):
    mat = loadmat(filename + ".mat")
    image = np.array(mat[matname])

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

    mat = {matname : image}
    savemat(filename + "_pp.mat", mat)

def tif2mat(folder_path, all_bands_save=False):

    images_path = os.listdir(folder_path)
    sar_image_path = (folder_path + ([x for x in images_path if "CSK" in x])[0])[:-4]

    sar_file = rio.open(sar_image_path+".tif")
    sar_data = sar_file.read()
    sar_file.close()
    mat_sar = {'CSK' : sar_data}
    savemat(sar_image_path + ".mat", mat_sar)
    postprocess(sar_image_path, "CSK", True)


    VNIR_image_path = (folder_path + ([x for x in images_path if "VNIR" in x])[0])[:-4]
    SWIR_image_path = (folder_path + ([x for x in images_path if "SWIR" in x])[0])[:-4]

    VNIR_data = rio.open(VNIR_image_path + ".tif") 
    VNIR_image = VNIR_data.read()
    VNIR_data.close()
    mat_vnir = {'VNIR' : VNIR_image}
    savemat(VNIR_image_path + ".mat", mat_vnir)
    postprocess(VNIR_image_path, "VNIR")

    SWIR_data = rio.open(SWIR_image_path + ".tif") 
    SWIR_image = SWIR_data.read()
    SWIR_data.close()
    mat_SWIR = {'SWIR' : SWIR_image}
    savemat(SWIR_image_path + ".mat", mat_SWIR)
    postprocess(SWIR_image_path, "SWIR")

    if (all_bands_save):
        all_band_raster = np.concatenate((VNIR_image,SWIR_image), axis=0)
        # all_band_image = np.reshape(all_band_image,(all_band_image.shape[1],all_band_image.shape[2],all_band_image.shape[0])) #reshape as image
        # all_band_image = np.moveaxis(all_band_raster,0,-1)
        mat_all_PRISMA = {'PRISMA' : all_band_raster}
        savemat(VNIR_image_path.replace("VNIR","Whole") + ".mat", mat_all_PRISMA)
        postprocess(VNIR_image_path.replace("VNIR","Whole"), "PRISMA")

def cut_data_window(input_path):
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
    with rio.open(input_path, 'r') as src:          
        image = src.read()
    with rio.open(profile_path, 'r') as src:          
        profile = src.profile
    with rio.open(output_path, 'w', **profile) as dst:          
        dst.write(image)

def georeference_png_file(profile_sample, results_path):
    
    # profile_sample might be a profile or the path to an image with a profile to sample (NOT TESTED)
    if isinstance(profile_sample, str):
        with rio.open(profile_sample, 'r') as src:          
            profile = src.profile
    else:
        profile = profile_sample
        
    confusion_image = Image.open(results_path,mode='r')

    np_results = np.reshape(np.array(confusion_image.getdata(band=1)),(1, confusion_image.height, confusion_image.width)) / 255

    with rio.open(results_path[:-4] + "_geo.tif", 'w', **profile) as dst:          
        dst.write(np_results)

def confusion_matrix(ground_truth_path, predicted_path):
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


if __name__ == "__main__":
    
    folder_name = "././data/CSK_after/"
    tif2mat(folder_name, True)