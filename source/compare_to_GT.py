import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio as rio
import sklearn.metrics
import math
import scipy
from skimage.filters import threshold_otsu
import pyperclip

paperclip=''

def compare(cm_png_path, gt_tiff_path, correct_gt=False, mask=None):
    # confusion_maps_png_path = "./Results/X-Net/Danubio_2/Confusion_map.png"
    confusion_maps_png_path = cm_png_path
    confusion_map_image = Image.open(confusion_maps_png_path)
    confusion_map = np.array(confusion_map_image)[:,:,1]

    # GT_path = "./data/Danubio/inspire_changes_rasterized_clipped_stacked_2.tif"
    GT_path = gt_tiff_path
    with rio.open(GT_path) as src:
        ground_truth = src.read().squeeze()

    if (correct_gt):
        ground_truth +=1
        ground_truth[ground_truth==ground_truth.min()]=0

    if mask is not None:
        ground_truth*= mask

    ground_truth_flat = np.reshape(ground_truth,(-1))
    ground_truth_filt = ground_truth_flat[ground_truth_flat!=0]

    confusion_map_flat = np.reshape(confusion_map,(-1))
    confusion_map_filt = np.array(confusion_map_flat[ground_truth_flat!=0] > 0, dtype='int')

    ground_truth_filt -= 1
    ground_truth_filt = np.array(ground_truth_filt, dtype='int')

    # print(confusion_map_filt.shape,ground_truth_filt.shape, np.unique(confusion_map_filt), np.unique(ground_truth_filt))
    # confusion_map_filt[:] = 1

    # confusion_matrix = sklearn.metrics.confusion_matrix(y_true = ground_truth_filt, y_pred = confusion_map_filt)

    # print(confusion_matrix @ [1,1])
    # print([1,1] @ confusion_matrix )

    accuracy = sklearn.metrics.accuracy_score(y_true = ground_truth_filt, y_pred = confusion_map_filt)
    precision_pos = sklearn.metrics.precision_score(y_true = ground_truth_filt, y_pred = confusion_map_filt, pos_label=1)
    precision_neg = sklearn.metrics.precision_score(y_true = ground_truth_filt, y_pred = confusion_map_filt, pos_label=0)
    # AUC = sklearn.metrics.roc_auc_score(y_true = ground_truth_filt, y_score = confusion_map_filt)
    recall_1 = sklearn.metrics.recall_score(y_true = ground_truth_filt, y_pred = confusion_map_filt, pos_label=1)
    recall_0 = sklearn.metrics.recall_score(y_true = ground_truth_filt, y_pred = confusion_map_filt, pos_label=0)
    kappa = sklearn.metrics.cohen_kappa_score(y1 = ground_truth_filt, y2 = confusion_map_filt)

    scores = {
        'precision_1':round(precision_pos,3),
        'precision_0':round(precision_neg,3),
        'sensitivity':round(recall_1,3),
        'specificity':round(recall_0,3),
        'G-Mean':round(math.sqrt(recall_0*recall_1),3),
        # 'AUC': AUC
        'accuracy':round(accuracy,3),
        # 'confusion_matrix':confusion_matrix
        'kappa':round(kappa, 3)
    }
    # paperclip+=scores['precision_1'] + '\t' + scores['precision_0'] + '\t' +scores['sensitivity'] + '\t' + scores['specificity'] + '\t' + scores['G-Mean'] + '\t' + scores['accuracy'] + '\n'

    return scores

def compare_two_png(cm_png_path, gt_png_path, correct_gt=False, mask=None, overlap=False):
    confusion_maps_png_path = cm_png_path
    confusion_map_image = Image.open(confusion_maps_png_path)
    confusion_map = np.array(confusion_map_image)[:,:,1]

    ground_truth_image = Image.open(gt_png_path)
    ground_truth = np.array(ground_truth_image)

    if mask is not None:
        ground_truth*= mask

    ground_truth_flat = np.reshape(ground_truth / 255,(-1))

    confusion_map_flat = np.reshape(confusion_map / 255,(-1))

    # print(confusion_map_filt.shape,ground_truth_filt.shape, np.unique(confusion_map_filt), np.unique(ground_truth_filt))
    # confusion_map_filt[:] = 1

    # confusion_matrix = sklearn.metrics.confusion_matrix(y_true = ground_truth_filt, y_pred = confusion_map_filt)

    # print(confusion_matrix @ [1,1])
    # print([1,1] @ confusion_matrix )

    accuracy = sklearn.metrics.accuracy_score(y_true = ground_truth_flat, y_pred = confusion_map_flat)
    precision_pos = sklearn.metrics.precision_score(y_true = ground_truth_flat, y_pred = confusion_map_flat, pos_label=1)
    precision_neg = sklearn.metrics.precision_score(y_true = ground_truth_flat, y_pred = confusion_map_flat, pos_label=0)
    # AUC = sklearn.metrics.roc_auc_score(y_true = ground_truth_flat, y_score = confusion_map_flat)
    recall_1 = sklearn.metrics.recall_score(y_true = ground_truth_flat, y_pred = confusion_map_flat, pos_label=1)
    recall_0 = sklearn.metrics.recall_score(y_true = ground_truth_flat, y_pred = confusion_map_flat, pos_label=0)
    kappa = sklearn.metrics.cohen_kappa_score(y1 = ground_truth_flat, y2 = confusion_map_flat)

    scores = {
        'precision_1':round(precision_pos,3),
        'precision_0':round(precision_neg,3),
        'sensitivity':round(recall_1,3),
        'specificity':round(recall_0,3),
        'G-Mean':round(math.sqrt(recall_0*recall_1),3),
        # 'AUC': AUC
        'accuracy':round(accuracy,3),
        'kappa':round(kappa, 3)
        # 'confusion_matrix':confusion_matrix
    }

    # paperclip+=scores['precision_1'] + '\t' + scores['precision_0'] + '\t' +scores['sensitivity'] + '\t' + scores['specificity'] + '\t' + scores['G-Mean'] + '\t' + scores['accuracy'] + '\n'

    if overlap:
        fig, ax = plt.subplots(2,1)
        # display_img = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3))
        # display_img[np.array(np.bitwise_and(confusion_map, ground_truth) / 255, dtype=bool),:] = [1,1,1]
        # display_img[np.array(np.bitwise_and(~confusion_map, ~ground_truth) / 255, dtype=bool),:] = [0,0,0]
        # display_img[np.array(np.bitwise_and(confusion_map, ~ground_truth) / 255, dtype=bool),:] = [0,1,0]
        # display_img[np.array(np.bitwise_and(~confusion_map, ground_truth) / 255, dtype=bool),:] = [1,0,0]
        # fig.imshow(ground_truth * 255)
        ax[0].imshow(ground_truth * 255)
        ax[1].imshow(confusion_map * 255)
        plt.show()

        


    return scores


if __name__ == '__main__':
    # gt_path = "./data/Danubio/inspire_changes_rasterized_clipped_stacked_2.tif"
    # gt_path = "./data/Danubio/changes_clipped_v3.tif"
    gt_path = "./data/Danubio/HomogeneousCD.png"

    confusion_map_path_1 = "./Results/X-Net/Danubio_1ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_1, gt_path, correct_gt=True)
    print(f'1ch: {scores}')

    confusion_map_path_3 = "./Results/X-Net/Danubio_3ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_3, gt_path, correct_gt=True)
    print(f'3ch: {scores}')

    # confusion_map_path_5 = "./Results/X-Net/Danubio_5ch/Confusion_map.png" 
    # scores = compare(cm_png_path=confusion_map_path_5, gt_png_path=gt_path, correct_gt=True)
    # print(f'5ch: {scores}')

    confusion_map_path_6 = "./Results/X-Net/Danubio_6ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_6, gt_path, correct_gt=True)
    print(f'6ch: {scores}')

    confusion_map_path_9 = "./Results/X-Net/Danubio_9ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_9, gt_path, correct_gt=True)#, overlap=True)
    print(f'9ch: {scores}')

    confusion_map_path_9_s = "./Results/X-Net/Danubio_native/Change_map.png" 
    scores = compare_two_png(confusion_map_path_9_s, gt_path, correct_gt=True)#, overlap=True)
    print(f'9_native: {scores}')

    confusion_map_path_10 = "./Results/X-Net/Danubio_10ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_10, gt_path, correct_gt=True)#, overlap=True)
    print(f'10ch: {scores}')

    confusion_map_path_10 = "./Results/X-Net/Danubio_10ch/Affinity_CD_map.png" 
    scores = compare_two_png(confusion_map_path_10, gt_path, correct_gt=True)
    print(f'10ch: {scores}')

    # confusion_map_path_15 = "./Results/X-Net/Danubio_15ch/Confusion_map.png" 
    # scores = compare(cm_png_path=confusion_map_path_15, gt_png_path=gt_path, correct_gt=True)
    # print(f'15ch: {scores}')

    confusion_map_path_20 = "./Results/X-Net/Danubio_20ch/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_20, gt_path, correct_gt=True)
    print(f'20ch: {scores}')

    confusion_map_path_20 = "./Results/X-Net/Danubio/Affinity_CD_map.png" 
    scores = compare_two_png(confusion_map_path_20, gt_path, correct_gt=True)
    print(f'20ch_aff: {scores}')

    pyperclip.copy(paperclip)

    confusion_map_path_masked = "./Results_masked_2pr_521e/X-Net/Danubio/Confusion_map.png" 
    scores = compare_two_png(confusion_map_path_masked, gt_path, correct_gt=True)
    print(f'train masked(2%) test not masked: {scores}')

    train_mask_mat = scipy.io.loadmat("data/Danubio/changes_clipped_v3_masked_2percent.mat")
    train_mask = np.array(train_mask_mat["train_mask"], dtype=np.uint8)
    scores = compare_two_png(confusion_map_path_masked, gt_path, correct_gt=True, mask=train_mask)
    print(f'train masked(10%) test masked: {scores}')
