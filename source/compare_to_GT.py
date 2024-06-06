import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio as rio
import sklearn.metrics
import math
import scipy
from skimage.filters import threshold_otsu
# import pyperclip

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
    paperclip+=scores['precision_1'] + '\t' + scores['precision_0'] + '\t' +scores['sensitivity'] + '\t' + scores['specificity'] + '\t' + scores['G-Mean'] + '\t' + scores['accuracy'] + '\n'

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
        ground_truth = ground_truth[4:,:-3]
        confusion_map = confusion_map[4:,:-3]

        printed_img = np.zeros((confusion_map.shape[0], confusion_map.shape[1], 3))


        plt.imshow(confusion_map, cmap='gray')
        plt.show()

        plt.imshow(ground_truth, cmap='gray')
        plt.show()

        printed_img[np.array(np.bitwise_and(~confusion_map, ~ground_truth), dtype=bool),:] = [0,0,0]
        printed_img[np.array(np.bitwise_and(confusion_map, ground_truth), dtype=bool),:] = [1,1,1]
        printed_img[np.array(np.bitwise_and(confusion_map, ~ground_truth), dtype=bool),:] = [0,1,0]
        printed_img[np.array(np.bitwise_and(~confusion_map, ground_truth), dtype=bool),:] = [0,0,1]
        plt.imshow(printed_img)
        plt.show()


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

def compare_np(GT, prediction, mask=None, overlap=False, paperclip=''):
    if mask is not None:
        GT = GT[mask==0]
        prediction = prediction[mask==0]
    
    GT_flat = np.reshape(GT,(-1))
    prediction_flat = np.reshape(prediction,(-1))
    accuracy = sklearn.metrics.accuracy_score(y_true = GT_flat, y_pred = prediction_flat)
    precision_pos = sklearn.metrics.precision_score(y_true = GT_flat, y_pred = prediction_flat, pos_label=1)
    precision_neg = sklearn.metrics.precision_score(y_true = GT_flat, y_pred = prediction_flat, pos_label=0)
    # producer accuracy di change
    # 1 - producer accuracy di no change
    # AUC = sklearn.metrics.roc_auc_score(y_true = GT_flat, y_score = prediction_flat)
    recall_1 = sklearn.metrics.recall_score(y_true = GT_flat, y_pred = prediction_flat, pos_label=1)
    recall_0 = sklearn.metrics.recall_score(y_true = GT_flat, y_pred = prediction_flat, pos_label=0)
    kappa = sklearn.metrics.cohen_kappa_score(y1 = GT_flat, y2 = prediction_flat)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = GT_flat, y_pred = prediction_flat)
    # calculate false alarm rate= FP / (FP + TN)
    FAR = confusion_matrix[0,1] / (confusion_matrix[0,1] + confusion_matrix[0,0])

    scores = {
        'precision_1(UA)':round(precision_pos,3),
        'precision_0':round(precision_neg,3),
        'sensitivity(PA)':round(recall_1,3),
        'specificity':round(recall_0,3),
        'G-Mean':round(math.sqrt(recall_0*recall_1),3),
        # 'AUC': AUC
        'accuracy':round(accuracy,3),
        'kappa':round(kappa, 3),
        'FAR':round(FAR, 3)
    }

    if overlap:

        ground_truth_c = GT.copy() + 1
        ground_truth_c[mask==1] = 0
        ground_truth_color = np.zeros((GT.shape[0], GT.shape[1], 3))
        ground_truth_color[ground_truth_c==2,:] = [1,1,1]
        ground_truth_color[ground_truth_c==1,:] = [0,0,0]
        ground_truth_color[ground_truth_c==0,:] = [1,0,0]


        prediction_c = prediction.copy() + 1
        prediction_c[mask==1] = 0
        prediction_color = np.zeros((GT.shape[0], GT.shape[1], 3))
        prediction_color[prediction_c==2,:] = [1,1,1]
        prediction_color[prediction_c==1,:] = [0,0,0]
        prediction_color[prediction_c==0,:] = [1,0,0]


        overlap = np.zeros((GT.shape[0], GT.shape[1], 3))
        overlap[np.array(np.bitwise_and(~prediction, ~GT), dtype=bool),:] = [0,0,0]
        overlap[np.array(np.bitwise_and(prediction, GT), dtype=bool),:] = [1,1,1]
        overlap[np.array(np.bitwise_and(prediction, ~GT), dtype=bool),:] = [0,1,0]
        overlap[np.array(np.bitwise_and(~prediction, GT), dtype=bool),:] = [0,0,1]
        overlap[mask==1,:] = [1,0,0]
        f, axarr = plt.subplots(1,3) 
        axarr[0].imshow(prediction_color)
        axarr[1].imshow(ground_truth_color)
        axarr[2].imshow(overlap)

        plt.show()

    
    # paperclip+=scores['precision_1'] + '\t' + scores['precision_0'] + '\t' +scores['sensitivity'] + '\t' + scores['specificity'] + '\t' + scores['G-Mean'] + '\t' + scores['accuracy'] + '\n'
    return scores

if __name__ == '__main__':
    # gt_path = "./data/Danubio/inspire_changes_rasterized_clipped_stacked_2.tif"
    # gt_path = "./data/Danubio/changes_clipped_v3.tif"
    # gt_path = "./data/Danubio/HomogeneousCD.png"

    # confusion_map_path_1 = "./Results/X-Net/Danubio_1ch/Confusion_map.png" 
    # scores = compare_two_png(confusion_map_path_1, gt_path, correct_gt=True)
    # print(f'1ch: {scores}')

    # confusion_map_path_3 = "./Results/X-Net/Danubio_3ch/Confusion_map.png" 
    # scores = compare_two_png(confusion_map_path_3, gt_path, correct_gt=True, overlap=True)
    # print(f'3ch: {scores}')

    # confusion_map_path_6 = "./Results/X-Net/Danubio_6ch/Confusion_map.png" 
    # scores = compare_two_png(confusion_map_path_6, gt_path, correct_gt=True)
    # print(f'6ch: {scores}')

    # confusion_map_path_9 = "./Results/X-Net/Danubio_9ch/Confusion_map.png" 
    # scores = compare_two_png(confusion_map_path_9, gt_path, correct_gt=True)#, overlap=True)
    # print(f'9ch: {scores}')

    # confusion_map_path_9_s = "./Results/X-Net/Danubio_native/Change_map.png" 
    # scores = compare_two_png(confusion_map_path_9_s, gt_path, correct_gt=True)#, overlap=True)
    # print(f'9_native: {scores}')

    # confusion_map_path_10 = "./Results/X-Net/Danubio_10ch/Confusion_map.png" 
    # scores = compare_two_png(confusion_map_path_10, gt_path, correct_gt=True)#, overlap=True)
    # print(f'10ch: {scores}')

    # confusion_map_path_10 = "./Results/X-Net/Danubio_10ch/Affinity_CD_map.png" 
    # scores = compare_two_png(confusion_map_path_10, gt_path, correct_gt=True)
    # print(f'10ch: {scores}')

    paperclip=''

    gt_path = "./data/E_R/ground_truth_flood.npy"
    gt_path_sentinel = "./data/E_R/Sentinel_CD_GT_pp.npy"
    gt_corrected = "./data/E_R/corrected_GT.npy"
    # gt_corrected = gt_path
    # gt_corrected = "./data/E_R/corrected_GT_flood.npy"
    # gt_corrected = "./data/E_R/GT_corrected_311023.npy"
    exclude_path = "./data/E_R/exclude.npy"
    ground_truth = np.load(gt_path).squeeze()
    ground_truth_sentinel = np.load(gt_path_sentinel).squeeze()
    ground_truth_corrected = np.load(gt_corrected).squeeze()
    exclude_gt = np.load(exclude_path).squeeze()
    # exclude_gt = None

    ALPHA_NOT_UPDATED = False
    ALPHA_UPDATED = True

    # confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_3ch/Change_map.png" 
    # confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
    # # scores = compare_np(ground_truth, confusion_map, mask=exclude_gt, overlap=True)
    # scores = compare_np(ground_truth_sentinel, confusion_map, overlap=True)
    # paperclip += scores.__str__() + '\n'
    # print("3ch:", scores)

    # confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_3ch/Change_map.png" 
    # confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
    # # scores = compare_np(ground_truth, confusion_map, mask=exclude_gt, overlap=True)
    # scores = compare_np(ground_truth_corrected, confusion_map, overlap=True)
    # paperclip += scores.__str__() + '\n'
    # print("3ch:", scores)


    # confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_3ch/Change_map.png" 
    # confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
    # scores = compare_np(ground_truth, confusion_map, mask=exclude_gt)
    # paperclip += scores.__str__() + '\n'
    # print("3ch:", scores)
    
    
    if(ALPHA_NOT_UPDATED):
        print('E_R_alpha_not_updated:')
        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_1ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("1ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_3ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("3ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_5ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("5ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_10ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("10ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_15ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("15ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_20ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("20ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_30ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("30ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_40ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("40ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_alpha_not_updated_50ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("50ch:", scores)

    if(ALPHA_UPDATED):
        print('E_R_alpha_updated:')    

        confusion_map_path = "./Results/X-Net/E_R_1ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("1ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_3ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("3ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_try4_3ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("3ch check:", scores)

        confusion_map_path = "./Results/X-Net/E_R_5ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("5ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_10ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("10ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_15ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("15ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_20ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("20ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_30ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("30ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_40ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("40ch:", scores)

        confusion_map_path = "./Results/X-Net/E_R_50ch/Change_map.png" 
        confusion_map = np.array(np.array(Image.open(confusion_map_path))[:,:,1]>1, dtype='int')
        scores = compare_np(ground_truth_corrected, confusion_map, mask=exclude_gt)
        paperclip += scores.__str__() + '\n'
        print("50ch:", scores)
    
    
    paperclip = paperclip.replace(',','\t')
    paperclip = paperclip.replace(':','\t')
    paperclip = paperclip.replace('{','')
    paperclip = paperclip.replace('}','')
    paperclip = paperclip.replace('\'','')
    paperclip = paperclip.replace('\"','')
    a=False
    # pyperclip.copy(paperclip)