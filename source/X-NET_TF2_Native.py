import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import scipy.io
import os
import sys
import time
from tqdm import tqdm

from PIL import ImageOps
from PIL import Image
from skimage.filters import threshold_otsu
import sklearn.metrics as mt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dropout, LeakyReLU

from skimage.measure import block_reduce
from scipy.ndimage import zoom
import argparse
import math
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
from pdb import set_trace as bp
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=0, type=int)
parser.add_argument("--tran", default=3.0, type=float)
parser.add_argument("--cycle", default=2.0, type=float)
args = parser.parse_args()

DATASET = args.dataset
PRE_TRAIN = 0 #binary
TRAIN = 1
NAME_DATASET = ["Danubio"]
USE_PATCHES = False  # use image as whole or in patches

LEARNING_RATE = 10e-5
EPOCHS = 240
MAX_BATCHES = 10
BATCH_SIZE = 10
PATCH_SIZE = 100


PATCH_SIZE_AFF = 40
PATCH_STRIDE_AFF = PATCH_SIZE_AFF // 4
BATCH_SIZE_AFF = 50
ZERO_PAD = 0
AFFINITY_RESOLUTIONS = 3

W_REG = 0.001
W_TRAN = args.tran
W_CYCLE = args.cycle
MAX_GRAD_NORM = 1.0 
DROP_PROB = 0.2
ALPHA_LEAKY = 0.3

nf1 = 100
nf2 = 50
nf3 = 20
nf4 = 10

fs1 = 3
fs2 = 3
fs3 = 3
fs4 = 3

if DATASET == 0:
    nc1 = 1
    nc2 = 9
else:
    print("Wrong dataset")
    exit()
specs_X_to_Y = [
    [nc1, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, nc2, fs4, 1],
]

specs_Y_to_X = [
    [nc2, nf1, fs1, 1],
    [nf1, nf2, fs2, 1],
    [nf2, nf3, fs3, 1],
    [nf3, nc1, fs4, 1],
]

class cross_network(object):

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.model = "././models/X-Net/"
        if not os.path.exists(self.model):
            os.makedirs(self.model)
        self.model_path_X_to_Y = self.model +  NAME_DATASET[DATASET] + "_X_to_Y.h5"
        self.model_path_Y_to_X = self.model +  NAME_DATASET[DATASET] + "_Y_to_X.h5"
        self.folder = "././Results/X-Net/" + NAME_DATASET[DATASET] + "_aff_40/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        net_X_to_Y = Sequential()
        for i in range(len(specs_X_to_Y)):
            net_X_to_Y.add(Conv2D(filters=specs_X_to_Y[i][1], kernel_size=specs_X_to_Y[i][2], 
                                input_shape=[None, None, nc1], padding="same"))
            if i < len(specs_X_to_Y) - 1:
                net_X_to_Y.add(LeakyReLU(ALPHA_LEAKY))            
                net_X_to_Y.add(Dropout(DROP_PROB))
            else:
                net_X_to_Y.add(Activation('tanh'))    

        net_Y_to_X = Sequential()
        for i in range(len(specs_Y_to_X)):
            net_Y_to_X.add(Conv2D(filters=specs_Y_to_X[i][1], kernel_size=specs_Y_to_X[i][2], 
                                input_shape=[None, None, nc2], padding="same"))
            if i < len(specs_Y_to_X) - 1:
                net_Y_to_X.add(LeakyReLU(ALPHA_LEAKY))
                net_Y_to_X.add(Dropout(DROP_PROB))
            else:
                net_Y_to_X.add(Activation('tanh'))

        self.net_X_to_Y = net_X_to_Y
        self.net_Y_to_X = net_Y_to_X

    def data_augmentation(self):
        batch_x = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc1])
        batch_y = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc2])
        batch_a = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            rotation = np.random.randint(4)
            a = np.random.randint(self.t1.shape[0] - PATCH_SIZE)
            b = a + PATCH_SIZE
            c = np.random.randint(self.t1.shape[1] - PATCH_SIZE)
            d = c + PATCH_SIZE
            batch_x[i] = np.rot90(self.t1[a:b, c:d, :], rotation)
            batch_y[i] = np.rot90(self.t2[a:b, c:d, :], rotation)
            batch_a[i] = np.rot90(self.Alpha[a:b, c:d, np.newaxis], rotation)
            if np.random.randint(2):
                batch_x[i] = np.flipud(batch_x[i])
                batch_y[i] = np.flipud(batch_y[i])
                batch_a[i] = np.flipud(batch_a[i])

        return batch_x, batch_y, batch_a

    def idx_patches(self, array):
        i, j = 0, 0
        idx = []
        while i + PATCH_SIZE_AFF <= array.shape[0]:
            idx.append([i, j])
            j += PATCH_STRIDE_AFF
            if j + PATCH_SIZE_AFF > array.shape[1]:
                i += PATCH_STRIDE_AFF
                j = 0
        return np.array(idx)

    def from_idx_to_patches(self, array, idx):
        res = []
        if ZERO_PAD == 0:
            end = None
        else:
            end = -ZERO_PAD
        for k in range(idx.shape[0]):
            i = idx[k, 0]
            j = i + PATCH_SIZE_AFF
            l = idx[k, 1]
            m = l + PATCH_SIZE_AFF
            sz = PATCH_SIZE_AFF + 2 * ZERO_PAD
            padded_array = np.zeros((sz, sz) + array.shape[2:])
            padded_array[ZERO_PAD:end, ZERO_PAD:end, ...] = array[i:j, l:m, ...]
            res.append(padded_array)
        return np.array(res)

    def save_image(self, array, subfolder):
        img = Image.fromarray(array.astype("uint8"))
        if subfolder.find("Aff") != -1 or subfolder.find("d") != -1:
            img = ImageOps.equalize(img, mask=None)
        img = img.convert("RGB")
        img.save(self.folder + subfolder)

    def filtering(self, d):
        d = d[..., np.newaxis]
        d = np.concatenate((d, 1.0 - d), axis=2)
        W = np.size(d, 0)
        H = np.size(d, 1)
        stack = np.concatenate((self.t1, self.t2), axis=2)
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

    def remove_borders(self, x):
        if PATCH_STRIDE_AFF != 1:
            s1 = x.shape[0]
            s2 = x.shape[1]
            remove_along_dim_1 = (s1 - PATCH_SIZE_AFF) % PATCH_STRIDE_AFF
            remove_along_dim_2 = (s2 - PATCH_SIZE_AFF) % PATCH_STRIDE_AFF
            up = remove_along_dim_1 // 2
            down = up - remove_along_dim_1
            if down == 0:
                down = None
            left = remove_along_dim_2 // 2
            right = left - remove_along_dim_2
            if right == 0:
                right = None
            x = x[up:down, left:right]
        return x

    @tf.function
    def affinity(self, x, y):
        x_1 = tf.expand_dims(tf.reshape(x, [-1, PATCH_SIZE_AFF ** 2, nc1]), 2)
        x_2 = tf.expand_dims(tf.reshape(x, [-1, PATCH_SIZE_AFF ** 2, nc1]), 1)
        A_x = tf.norm(tensor=x_1 - x_2, axis=-1)
        del x_1, x_2
        k_x = tf.nn.top_k(A_x, k=A_x.shape[-1]).values
        k_x = tf.reduce_mean(input_tensor=k_x[:, :, -(PATCH_SIZE_AFF ** 2) // 2], axis=1)
        y_1 = tf.expand_dims(tf.reshape(y, [-1, PATCH_SIZE_AFF ** 2, nc2]), 2)
        y_2 = tf.expand_dims(tf.reshape(y, [-1, PATCH_SIZE_AFF ** 2, nc2]), 1)
        A_y = tf.norm(tensor=y_1 - y_2, axis=-1)
        del y_1, y_2
        k_y = tf.nn.top_k(A_y, k=A_y.shape[-1]).values
        k_y = tf.reduce_mean(input_tensor=k_y[:, :, -(PATCH_SIZE_AFF ** 2) // 2], axis=-1)
        k_x = tf.reshape(k_x, (-1, 1, 1))
        k_y = tf.reshape(k_y, (-1, 1, 1))
        A_x = tf.exp(-(tf.divide(A_x, k_x) ** 2))
        A_y = tf.exp(-(tf.divide(A_y, k_y) ** 2))
        D = tf.reshape(
            tf.reduce_mean(input_tensor=tf.abs(A_x - A_y), axis=-1),
            [-1, PATCH_SIZE_AFF, PATCH_SIZE_AFF],
        )
        return D
    
    def pre_train(self):
        sizes = AFFINITY_RESOLUTIONS
        D = np.zeros((self.t1.shape[0], self.t1.shape[1], sizes))
        for i in range(-1, sizes - 1):
            if i < 0:
                x_resize = zoom(self.t1, (2 ** (-i), 2 ** (-i), 1))
                y_resize = zoom(self.t2, (2 ** (-i), 2 ** (-i), 1))
            else:
                x_resize = block_reduce(self.t1, (2 ** i, 2 ** i, 1), np.mean)
                y_resize = block_reduce(self.t2, (2 ** i, 2 ** i, 1), np.mean)
            d = self.get_affinities(
                self.remove_borders(x_resize), self.remove_borders(y_resize)
            )
            d = np.array(Image.fromarray(d).resize((D.shape[1], D.shape[0])))
            self.save_image(255.0 * d, "Aff_" + str(i) + ".png")
            D[..., i] = d
        return np.mean(D, axis=-1)
    
    def get_affinities(self, x, y):
        aff = np.zeros((x.shape[0], x.shape[1]))
        covers = np.copy(aff)
        idx = self.idx_patches(aff)
        runs = idx.shape[0] // BATCH_SIZE_AFF
        print("Runs: {}".format(runs))
        print("Leftovers: {}".format(idx.shape[0] % BATCH_SIZE_AFF))
        time1 = time.time()
        done_runs = 0
        if idx.shape[0] % BATCH_SIZE != 0:
            runs += 1
        for i in tqdm(range(runs)):
            temp_idx = idx[:BATCH_SIZE_AFF]
            batch_t1 = self.from_idx_to_patches(x, temp_idx)
            batch_t2 = self.from_idx_to_patches(y, temp_idx)
            Aff = self.affinity(batch_t1, batch_t2)
            
            for i in range(temp_idx.shape[0]):
                p = temp_idx[i, 0]
                q = p + PATCH_SIZE_AFF
                r = temp_idx[i, 1]
                s = r + PATCH_SIZE_AFF
                aff[p:q, r:s] += Aff[i]
                covers[p:q, r:s] += 1
            idx = idx[BATCH_SIZE_AFF:]
        aff = np.divide(aff, covers)
        return aff

    def train_model(self):
        prior_path = "././data/" + NAME_DATASET[DATASET] + "/change-prior_aff_40.mat"
        prior_name = "aff"# + str(self.t1.shape[0]) + str(self.t1.shape[1])
        try:
            if PRE_TRAIN:
                raise Exception("Forcing prior computation")
            self.Alpha = np.squeeze(scipy.io.loadmat(prior_path)[prior_name])
            if np.isnan(self.Alpha).any():
                print("Nan found in affinity")
                self.Alpha[np.isnan(self.Alpha)] = 1
        except Exception as exc:
            print(exc)
            print("Prior under evaluation")
            self.Alpha = self.pre_train()
            if np.isnan(self.Alpha).any():
                print("Nan found in affinity")
                self.Alpha[np.isnan(self.Alpha)] = 1
            scipy.io.savemat(prior_path, {"aff": self.Alpha})
            
        self.save_image(255.0 * self.Alpha, "alpha.png")

        datetime_str = str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute) + str(datetime.now().second) 
        writer = tf.summary.create_file_writer("././logs/train/X-Net" + datetime_str)
        #check how to load and save network parameters

        if TRAIN:            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, 10000, 0.96, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            mse = tf.keras.losses.MeanSquaredError()
            mse_tran = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            try:
                for epoch in tqdm(range(EPOCHS)):

                    losses = tf.zeros(6)

                    for _ in range(MAX_BATCHES):
                        x_patch, y_patch, aff_patch = self.data_augmentation()
                        summary = self.train_step(x_patch, y_patch, aff_patch, optimizer, mse, mse_tran)
                        loss = tf.convert_to_tensor(summary, dtype=float)
                        losses += loss

                    losses /= MAX_BATCHES
                    # writer.add_summary(summary, epoch)
                    with writer.as_default():
                        tf.summary.scalar('X_tot', losses[0], step=epoch)
                        tf.summary.scalar('Y_tot', losses[1], step=epoch)
                        tf.summary.scalar('X_Cycle', losses[2], step=epoch)
                        tf.summary.scalar('Y_Cycle', losses[3], step=epoch)
                        tf.summary.scalar('X_to_Y', losses[4], step=epoch)
                        tf.summary.scalar('Y_to_X', losses[5], step=epoch)

                    if epoch % (EPOCHS // 3) == 0 and epoch > 0:
                        self.Alpha = self.evaluate(save=False) #WHY               
                
                self.net_X_to_Y.save(self.model_path_X_to_Y)
                self.net_Y_to_X.save(self.model_path_Y_to_X)
            except KeyboardInterrupt:
                print("\ntraining interrupted")
        else:
            self.net_X_to_Y = tf.keras.models.load_model(self.model_path_X_to_Y)
            self.net_Y_to_X = tf.keras.models.load_model(self.model_path_Y_to_X)
        
        _ = self.evaluate(save=True)

    @tf.function
    def train_step(self, x_patch, y_patch, aff_patch, optimizer, mse, mse_tran):
        summary = [None] *6
        with tf.GradientTape() as tape:
            y_hat = self.net_X_to_Y(x_patch, training=True)
            x_hat = self.net_Y_to_X(y_patch, training=True)
            y_tilde = self.net_X_to_Y(x_hat, training=True)
            x_tilde = self.net_Y_to_X(y_hat, training=True)

            loss_Cycle_Y = mse(y_patch,y_tilde)
            loss_Cycle_X = mse(x_patch,x_tilde)

            loss_X_to_Y = mse_tran(y_patch,y_hat,1-aff_patch)  
            loss_Y_to_X = mse_tran(x_patch,x_hat,1-aff_patch)

            loss_reg_X = 0
            for tf_var in self.net_X_to_Y.weights:
                loss_reg_X += tf.reduce_mean(input_tensor=tf.nn.l2_loss(tf_var))
            
            loss_reg_Y = 0
            for tf_var in self.net_Y_to_X.weights:
                loss_reg_Y += tf.reduce_mean(input_tensor=tf.nn.l2_loss(tf_var))

            tot_loss_X = (
                W_CYCLE * loss_Cycle_X
                + W_REG * loss_reg_X
                + W_TRAN * loss_X_to_Y
            )

            tot_loss_Y = (
                W_CYCLE * loss_Cycle_Y
                + W_REG * loss_reg_Y
                + W_TRAN * loss_Y_to_X
            )

            tot_loss = tot_loss_X + tot_loss_Y         
            summary[0] = tf.reduce_mean(tot_loss_X)
            summary[1] = tf.reduce_mean(tot_loss_Y)
            summary[2] = loss_Cycle_X
            summary[3] = loss_Cycle_Y
            summary[4] = tf.reduce_mean(loss_X_to_Y)
            summary[5] = tf.reduce_mean(loss_Y_to_X)

        trainable_variables = self.net_X_to_Y.trainable_variables
        trainable_variables.extend(self.net_Y_to_X.trainable_variables)
        grads = tape.gradient(tot_loss, trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer.apply_gradients(zip(clipped_grads, trainable_variables))

        return summary

    def transform_images(self):
        if USE_PATCHES:
            x_hat = np.zeros(self.t1.shape)
            y_hat = np.zeros(self.t2.shape)            
            covers = np.zeros(self.mask.shape)
            idx = self.idx_patches(covers)
            runs = idx.shape[0] // BATCH_SIZE_AFF
            print("Runs: {}".format(runs))
            print("Leftovers: {}".format(idx.shape[0] % BATCH_SIZE_AFF))
            if self.idx.shape[0] % BATCH_SIZE != 0:
                runs += 1
            for i in tqdm(range(runs)):
                temp_idx = idx[0:BATCH_SIZE_AFF, :]
                batch_t1 = self.from_idx_to_patches(self.t1, temp_idx)
                batch_t2 = self.from_idx_to_patches(self.t2, temp_idx)
                temp2 = self.net_Y_to_X(batch_t2)
                temp1 = self.net_X_to_Y(batch_t1)
                for i in range(temp_idx.shape[0]):
                    p = temp_idx[i, 0]
                    q = p + PATCH_SIZE_AFF
                    r = temp_idx[i, 1]
                    s = q + PATCH_SIZE_AFF
                    y_hat[p:q, r:s, :] += temp1[i]
                    x_hat[p:q, r:s, :] += temp2[i]
                    covers[p:q, r:s] += 1
                del temp_idx
                idx = idx[BATCH_SIZE_AFF:]

            x_hat = np.divide(x_hat, covers[..., np.newaxis])
            y_hat = np.divide(y_hat, covers[..., np.newaxis])
        else:
            t1 = self.t1[np.newaxis, ...]
            t2 = self.t2[np.newaxis, ...]
            x_hat = self.net_Y_to_X(t2)
            y_hat = self.net_X_to_Y(t1)
            y_hat = y_hat[0]
            x_hat = x_hat[0]

        return x_hat, y_hat

    def evaluate(self, save):
        x_hat, y_hat = self.transform_images()#check if there are NaNs
        d_x = self.t1 - x_hat
        d_y = self.t2 - y_hat
        d_x = np.linalg.norm(d_x, 2, -1)
        d_y = np.linalg.norm(d_y, 2, -1)

        d_x[d_x > np.mean(d_x) + 3.0 * np.std(d_x)] = np.mean(d_x) + 3.0 * np.std(d_x)
        d_y[d_y > np.mean(d_y) + 3.0 * np.std(d_y)] = np.mean(d_y) + 3.0 * np.std(d_y)
        d_x = d_x / np.max(d_x)
        d_y = d_y / np.max(d_y)
        d = (d_x + d_y) / 2.0

        heatmap = self.filtering(d)
        otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))        
        CD_map = heatmap >= otsu

        aff_heatmap = self.filtering(self.Alpha)
        otsu_aff = threshold_otsu(aff_heatmap)
        aff_CD_map = aff_heatmap >= otsu_aff

        if save:
            self.save_image(255.0 * d_x, "d_x.png")
            self.save_image(255.0 * d_y, "d_y.png")
            self.save_image(255.0 * d, "d.png")
            self.save_image(255.0 * heatmap, "d_filtered.png")
            self.save_image(255.0 * CD_map, "Change_map.png")
            self.save_image(255.0 * aff_CD_map, "Affinity_CD_map.png")
            if nc1 > 3:
                self.save_image(255.0 * (self.t1[..., 1:4] + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (np.asarray(x_hat[..., 1:4]) + 1.0) / 2.0, "x_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t1) + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (np.squeeze(x_hat) + 1.0) / 2.0, "x_hat.png")
            if nc2 > 3:
                self.save_image(255.0 * (self.t2[..., 3:6] + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (np.asarray(y_hat[..., 3:6]) + 1.0) / 2.0, "y_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t2) + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (np.squeeze(y_hat) + 1.0) / 2.0, "y_hat.png")

        return d

def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2 - 1
    return image

def run_model():
    if DATASET == 0:
        mat = scipy.io.loadmat("././data/Danubio/CSKS2_GTC_B_HI_0B_HH_RD_SF_20200908_clipped_resamp_pp.mat")
        t1 = np.array(mat["CSK"], dtype=float)
        t1 = np.expand_dims(t1.squeeze(),-1)
        mat2 = scipy.io.loadmat("././data/Danubio/PRISMA_9ch_pp.mat")
        t2 = np.array(mat2["PRISMA"], dtype=float)
    else:
        print("Wrong data set")
        exit()
    del mat
    time1 = time.time()
    cross_net = cross_network(t1, t2)
    cross_net.train_model()
    return time.time() - time1

if __name__ == "__main__":
    times = run_model()
    print(times)
