import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import scipy.io
import os
import sys
import time
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import ImageOps
from PIL import Image
from skimage.filters import threshold_otsu
import sklearn.metrics as mt
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

config = tf.compat.v1.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
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
parser.add_argument("--dataset", default=3, type=int)
parser.add_argument("--tran", default=3.0, type=float)
parser.add_argument("--cycle", default=2.0, type=float)
parser.add_argument("--abl_alpha", default=False, type=bool)
args = parser.parse_args()

DATASET = args.dataset
PRE_TRAIN = 0 #binary
TRAIN = 1
NAME_DATASET = ["Texas", "California", "Shuguang", "Danubio"]
USE_PATCHES = 0  # DATASET
UNSUPERVISED = True

LEARNING_RATE = 10e-5
EPOCHS = 240
# EPOCHS = 512
MAX_BATCHES = 10
BATCH_SIZE = 10
PATCH_SIZE = 100


PATCH_SIZE_AFF = 20
PATCH_STRIDE_AFF = PATCH_SIZE_AFF // 4
BATCH_SIZE_AFF = 100
ZERO_PAD = 0

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

if DATASET == 1:
    nc1 = 11
    nc2 = 3
elif DATASET == 3:
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


class network(object):
    def __init__(self, specs, x=None, train=None):
        self.specs = specs
        self.n_in = specs[0][0]
        self.temp = None
        self.layer_input = None
        if x is None:
            self.x = tf.compat.v1.placeholder("float", [None, None, None, self.n_in], name="x")
        else:
            self.x = x
        if train is None:
            self.train = tf.compat.v1.placeholder(tf.bool, name="train")
        else:
            self.train = train
        with tf.compat.v1.name_scope("Parameters_Network"):
            with tf.compat.v1.name_scope("Weights"):
                self.weights = {}
                for i, l in enumerate(self.specs):
                    if len(l) > 2:
                        self.weights["layer_{}".format(i + 1)] = tf.Variable(
                            tf.random.truncated_normal(
                                [l[2], l[2], l[0], l[1]],
                                stddev=math.sqrt(2 / (l[2] * l[2] * l[0])),
                            ),
                            name="w_{}".format(i + 1),
                        )
                    if len(l) == 2:
                        self.weights["layer_{}".format(i + 1)] = tf.Variable(
                            tf.random.truncated_normal(
                                [l[0], l[1]], stddev=math.sqrt(2 / (l[1] * l[1] * l[0]))
                            ),
                            name="w_{}".format(i + 1),
                        )

            with tf.compat.v1.name_scope("Biases"):
                self.biases = {}
                for i, l in enumerate(self.specs):
                    if l[1] != 0:
                        self.biases["layer_{}".format(i + 1)] = tf.Variable(
                            tf.zeros(l[1]), name="b_{}".format(i + 1)
                        )

        self.layers()
        self.output = self.network()

    def train_var(self):
        val = [val for _, val in self.weights.items()]
        val += [val for _, val in self.biases.items()]
        return val

    def layers(self):
        last = len(self.specs)
        for i, l in enumerate(self.specs):
            if len(l) > 2:
                exec("self.layer_" + str(i + 1) + " = self.make_conv(l,i,last)")
            elif len(l) == 2:
                exec("self.layer_" + str(i + 1) + " = self.make_fully_con(l,i,last)")

    def network(self, x=None):
        with tf.compat.v1.name_scope("Network"):
            if x is None:
                x = self.x#raroooo
            for i, _ in enumerate(self.specs):
                if i == 0:
                    layer_input = x
                with tf.compat.v1.name_scope("layer_{}".format(i + 1)) as scope:
                    exec("self.temp = self.layer_" + str(i + 1) + "(layer_input)")
                layer_input = self.temp
            return self.temp

    def make_conv(self, l, i, last):
        def _function(layer_input):
            sc = "layer_{}".format(i + 1)
            out = tf.nn.conv2d(
                input=layer_input, filters=self.weights[sc], strides=[1, l[3], l[3], 1], padding="SAME"
            )
            out = tf.nn.bias_add(out, self.biases[sc])
            # out = tf.layers.batch_normalization(out,training = self.train)
            # out = batch_norm_layer(out,self.train)
            if i != (last - 1):
                out = self.leaky_relu(out, ALPHA_LEAKY)
                out = tf.compat.v1.layers.dropout(out, rate=DROP_PROB, training=self.train)
            else:
                out = tf.nn.tanh(out)
            return out

        return _function

    def make_fully_con(self, l, i, last):
        def _function(layer_input):
            sc = "layer_{}".format(i + 1)
            out = tf.matmul(self.layer_input, self.weights[sc])
            out = tf.nn.bias_add(out, self.biases[sc])
            # out = tf.layers.batch_normalization(out,training = self.train)
            # out = batch_norm_layer(out,self.train)
            if i != (last - 1):
                out = self.leaky_relu(out, ALPHA_LEAKY)
                out = tf.compat.v1.layers.dropout(out, rate=DROP_PROB, training=self.train)
            else:
                out = tf.nn.sigmoid(out)
            return out

        return _function

    def leaky_relu(self, features, alpha=0.3, name="LeakyRelu"):
        with tf.compat.v1.name_scope(name, "LeakyRelu", [features, alpha]):
            features = tf.convert_to_tensor(value=features, name="features")
            alpha = tf.convert_to_tensor(value=alpha, name="alpha")
            return tf.maximum(alpha * features, features)


class cross_network(object):
    def __init__(self, t1, t2, mask, train_mask):
        self.t1 = t1
        self.t2 = t2
        self.mask = mask
        self.t_m = train_mask
        self.model = "models/X-Net/"
        if not os.path.exists(self.model):
            os.makedirs(self.model)
        self.model += NAME_DATASET[DATASET] + ".ckpt"
        self.folder = "Results/X-Net/" + NAME_DATASET[DATASET] + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            
        self.x = tf.compat.v1.placeholder("float", [None, None, None, nc1], name="x")
        self.y = tf.compat.v1.placeholder("float", [None, None, None, nc2], name="y")
        self.train = tf.compat.v1.placeholder(tf.bool, name="train")
        self.alpha = tf.compat.v1.placeholder("float", [None, None, None, 1], name="alpha")
        self.train_mask = tf.compat.v1.placeholder("float", [None, None, None, 1], name="train_mask")
        self.net_X_to_Y = network(specs_X_to_Y, self.x, self.train)
        self.net_Y_to_X = network(specs_Y_to_X, self.y, self.train)
        with tf.compat.v1.name_scope("X_to_Y"):
            self.Y_hat = self.net_X_to_Y.output

        with tf.compat.v1.name_scope("Y_to_X"):
            self.X_hat = self.net_Y_to_X.output

        with tf.compat.v1.name_scope("Cycle_X"):
            self.X_tilde = self.net_Y_to_X.network(self.net_X_to_Y.output)

        with tf.compat.v1.name_scope("Cycle_Y"):
            self.Y_tilde = self.net_X_to_Y.network(self.net_Y_to_X.output)

        with tf.compat.v1.name_scope("Loss"):
            self.parameters = tf.compat.v1.trainable_variables()
            self.loss_Cycle_X = tf.compat.v1.losses.mean_squared_error(
                labels=self.x * self.train_mask, predictions=self.X_tilde * self.train_mask
            )
            self.loss_Cycle_Y = tf.compat.v1.losses.mean_squared_error(
                labels=self.y * self.train_mask, predictions=self.Y_tilde * self.train_mask
            )
            self.loss_X_to_Y = tf.compat.v1.losses.mean_squared_error(
                labels=self.y * self.train_mask, predictions=self.Y_hat * self.train_mask, weights=1.0 - self.alpha
            )
            self.loss_Y_to_X = tf.compat.v1.losses.mean_squared_error(
                labels=self.x * self.train_mask, predictions=self.X_hat * self.train_mask, weights=1.0 - self.alpha
            )

            with tf.compat.v1.name_scope("L2_loss"):
                self.reg_loss_X = 0
                for tf_var in self.net_X_to_Y.train_var():
                    self.reg_loss_X += tf.reduce_mean(input_tensor=tf.nn.l2_loss(tf_var))
                self.reg_loss_Y = 0
                for tf_var in self.net_Y_to_X.train_var():
                    self.reg_loss_Y += tf.reduce_mean(input_tensor=tf.nn.l2_loss(tf_var))

            self.tot_loss_X = (
                W_CYCLE * self.loss_Cycle_X
                + W_REG * self.reg_loss_X
                + W_TRAN * self.loss_X_to_Y
            )

            self.tot_loss_Y = (
                W_CYCLE * self.loss_Cycle_Y
                + W_REG * self.reg_loss_Y
                + W_TRAN * self.loss_Y_to_X
            )

            self.tot_loss = self.tot_loss_X + self.tot_loss_Y
            global_step = tf.Variable(0, trainable=False)
            lr = tf.compat.v1.train.exponential_decay(
                LEARNING_RATE, global_step, 10000, 0.96, staircase=True
            )
            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            gradients = tf.gradients(ys=self.tot_loss, xs=self.parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
            self.update = optimizer.apply_gradients(
                zip(clipped_gradients, self.parameters), global_step
            )

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

    def data_augmentation(self):
        batch_x = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc1])
        batch_y = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc2])
        batch_a = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        batch_tm = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            rotation = np.random.randint(4)
            a = np.random.randint(self.t1.shape[0] - PATCH_SIZE)
            b = a + PATCH_SIZE
            c = np.random.randint(self.t1.shape[1] - PATCH_SIZE)
            d = c + PATCH_SIZE
            batch_x[i] = np.rot90(self.t1[a:b, c:d, :], rotation)
            batch_y[i] = np.rot90(self.t2[a:b, c:d, :], rotation)
            batch_a[i] = np.rot90(self.Alpha[a:b, c:d, np.newaxis], rotation)
            batch_tm[i] = np.rot90(self.t_m[a:b, c:d, np.newaxis], rotation)
            if np.random.randint(2):
                batch_x[i] = np.flipud(batch_x[i])
                batch_y[i] = np.flipud(batch_y[i])
                batch_a[i] = np.flipud(batch_a[i])
                batch_tm[i] = np.flipud(batch_tm[i])

        return batch_x, batch_y, batch_a, batch_tm

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
        # print("Filtering!")
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
        sizes = 3  ### ARBITRARY
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
            Aff = self.sess.run(
                self.affinity(self.x, self.y), {self.x: batch_t1, self.y: batch_t2}
            )
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
        saver = tf.compat.v1.train.Saver()

        with tf.compat.v1.name_scope("Tensorboard"):
        #     tf.compat.v1.summary.image(
        #         "X_Cycle",
        #         tf.cast((self.X_tilde[..., :] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "x",
        #         tf.cast((self.x[..., :] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "Y_Cycle",
        #         tf.cast((self.Y_tilde[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "y",
        #         tf.cast((self.y[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "X_hat",
        #         tf.cast((self.X_hat[..., :] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "Y_hat",
        #         tf.cast((self.Y_hat[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
        #         max_outputs=10,
        #     )
        #     tf.compat.v1.summary.image(
        #         "prior", tf.cast(self.alpha * 255, tf.uint8), max_outputs=10
        #     )
            tf.compat.v1.summary.scalar("X_Tot", self.tot_loss_X)
            tf.compat.v1.summary.scalar("X_Cycle", self.loss_Cycle_X)
            tf.compat.v1.summary.scalar("X_to_Y", self.loss_X_to_Y)
            tf.compat.v1.summary.scalar("Y_Tot", self.tot_loss_Y)
            tf.compat.v1.summary.scalar("Y_Cycle", self.loss_Cycle_Y)
            tf.compat.v1.summary.scalar("Y_to_X", self.loss_Y_to_X)

        with tf.compat.v1.Session(config=config) as self.sess:
            prior_path = "data/" + NAME_DATASET[DATASET] + "/change-prior.mat"
            prior_name = "aff" + str(self.mask.shape[0]) + str(self.mask.shape[1])
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
                scipy.io.savemat(prior_path, {prior_name: self.Alpha})
                
            self.save_image(255.0 * self.Alpha, "alpha.png")

            if args.abl_alpha:
                self.Alpha = np.random.rand(*self.Alpha.shape)
            datetime_str = str(datetime.now().year) +str(datetime.now().month) +str(datetime.now().day) +str(datetime.now().hour) +str(datetime.now().minute) +str(datetime.now().second) 
            writer = tf.compat.v1.summary.FileWriter("logs/train/X-Net" + datetime_str, graph=self.sess.graph)
            merged = tf.compat.v1.summary.merge_all()
            tf.compat.v1.global_variables_initializer().run()

            # Counting total number of parameters
            total_parameters = 0
            parameters = tf.compat.v1.trainable_variables()
            for variable in parameters:
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            # print("Total number of parameters : {}".format(total_parameters))

            if TRAIN:
                try:
                    for epoch in tqdm(range(EPOCHS)):
                        for p in range(MAX_BATCHES):
                            b_1, b_2, b_a, b_tm = self.data_augmentation()
                            summary, _ = self.sess.run(
                                [merged, self.update],
                                feed_dict={
                                    self.x: b_1,
                                    self.y: b_2,
                                    self.alpha: b_a,
                                    self.train: True,
                                    self.train_mask:b_tm
                                },
                            )
                        writer.add_summary(summary, epoch)
                        if epoch % (EPOCHS // 3) == 0 and epoch > 0:
                            self.Alpha = self.evaluate(save=False)
                    saver.save(self.sess, self.model)
                except KeyboardInterrupt:
                    print("\ntraining interrupted")
            else:
                saver.restore(self.sess, self.model)

            _ = self.evaluate(save=True)

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
                temp2, temp1 = self.sess.run(
                    [self.X_hat, self.Y_hat],
                    {self.x: batch_t1, self.y: batch_t2, self.train: False},
                )
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
            time2 = time.time()
            #print((time2 - time1) // 3600)
        else:
            t1 = self.t1[np.newaxis, ...]
            t2 = self.t2[np.newaxis, ...]
            x_hat, y_hat = self.sess.run(
                [self.X_hat, self.Y_hat], {self.x: t1, self.y: t2, self.train: False}
            )
            y_hat = y_hat[0]
            x_hat = x_hat[0]

        return x_hat, y_hat

    def evaluate(self, save):
        x_hat, y_hat = self.transform_images()#check why there are Nan's
        d_x = self.t1 - x_hat
        d_y = self.t2 - y_hat
        d_x = np.linalg.norm(d_x, 2, -1)
        d_y = np.linalg.norm(d_y, 2, -1)

        d_x[d_x > np.mean(d_x) + 3.0 * np.std(d_x)] = np.mean(d_x) + 3.0 * np.std(d_x)
        d_y[d_y > np.mean(d_y) + 3.0 * np.std(d_y)] = np.mean(d_y) + 3.0 * np.std(d_y)
        d_x = d_x / np.max(d_x)
        d_y = d_y / np.max(d_y)
        d = (d_x + d_y) / 2.0

        # AUC_b = mt.roc_auc_score(self.mask.flatten(), d.flatten())
        # # print("AUC_b: " + str(AUC_b))
        # otsu = threshold_otsu(d)
        # CD_map = d >= otsu
        # F1_Score_b = mt.f1_score(self.mask.flatten(), CD_map.flatten())
        # # print("F1_Score_b: " + str(F1_Score_b))
        # OA_b = mt.accuracy_score(self.mask.flatten(), CD_map.flatten())
        # # print("OA_b: " + str(OA_b))
        # KC_b = mt.cohen_kappa_score(self.mask.flatten(), CD_map.flatten())
        # # print("KC_b: " + str(KC_b))

        heatmap = self.filtering(d)
        otsu = threshold_otsu(heatmap)# local_otsu = otsu(heatmap, disk(15))
        # otsu = 
        CD_map = heatmap >= otsu  # CD_map = heatmap >= local_otsu

        aff_heatmap = self.filtering(self.Alpha)
        otsu_aff = threshold_otsu(aff_heatmap)
        aff_CD_map = aff_heatmap >= otsu_aff

        conf_map = np.zeros_like(CD_map)
        conf_map = np.tile(conf_map[..., np.newaxis], (1, 1, 3))
        conf_map[np.logical_and(self.mask, CD_map)] = [1, 1, 1]
        conf_map[np.logical_and(self.mask, np.logical_not(CD_map)), :] = [1, 0, 0]
        conf_map[np.logical_and(np.logical_not(self.mask), CD_map), :] = [0, 1, 0]

        AUC = mt.roc_auc_score(self.mask.flatten(), heatmap.flatten())
        AUPRC = mt.average_precision_score(self.mask.flatten(), heatmap.flatten())

        PREC_0 = mt.precision_score(self.mask.flatten(), CD_map.flatten(), pos_label=0)
        PREC_1 = mt.precision_score(self.mask.flatten(), CD_map.flatten())
        REC_0 = mt.recall_score(self.mask.flatten(), CD_map.flatten(), pos_label=0)
        REC_1 = mt.recall_score(self.mask.flatten(), CD_map.flatten())
        KC = mt.cohen_kappa_score(self.mask.flatten(), CD_map.flatten())
        [[TN, FP], [FN, TP]] = mt.confusion_matrix(
            self.mask.flatten(), CD_map.flatten()
        )
        self.evaluation = [TP, TN, FP, FN, PREC_0, REC_0, PREC_1, REC_1, KC, AUC, AUPRC]
        if save:
            self.save_image(255.0 * d_x, "d_x.png")
            self.save_image(255.0 * d_y, "d_y.png")
            self.save_image(255.0 * d, "d.png")
            self.save_image(255.0 * heatmap, "d_filtered.png")
            self.save_image(255.0 * conf_map, "Confusion_map.png")
            self.save_image(255.0 * aff_CD_map, "Affinity_CD_map.png")
            if nc1 > 3:
                self.save_image(255.0 * (self.t1[..., 1:4] + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (x_hat[..., 1:4] + 1.0) / 2.0, "x_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t1) + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (np.squeeze(x_hat) + 1.0) / 2.0, "x_hat.png")
            if nc2 > 3:
                self.save_image(255.0 * (self.t2[..., 3:6] + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (y_hat[..., 3:6] + 1.0) / 2.0, "y_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t2) + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (np.squeeze(y_hat) + 1.0) / 2.0, "y_hat.png")

        return d

def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2 - 1
    return image


def run_model():
    if DATASET == 1:
        mat = scipy.io.loadmat("./data/California/UiT_HCD_California_2017.mat")
        t1 = np.array(mat["t1_L8_clipped"], dtype=float)
        t2 = np.array(mat["logt2_clipped"], dtype=float)
        mask = np.array(mat["ROI"], dtype=bool)
        t1 = block_reduce(t1, (4, 4, 1), np.mean)
        t2 = block_reduce(t2, (4, 4, 1), np.mean)
        mask = block_reduce(mask, (4, 4), np.max)
    elif DATASET == 3:
        mat = scipy.io.loadmat("data/Danubio/CSK_20200908_synthetic.mat")
        t1 = np.array(mat["CSK"], dtype=float)
        mat2 = scipy.io.loadmat("data/Danubio/PRISMA_9ch_pp.mat")
        t2 = np.array(mat2["PRISMA"], dtype=float)
        # t2 = np.array(mat["CSK"], dtype=float) * 1.2
        # t1 = np.expand_dims(t1, axis = 2)
        # t2 = np.expand_dims(t2, axis = 2)
        # t1 = block_reduce(t1, (4, 4, 1), np.mean)
        # t2 = block_reduce(t2, (4, 4, 1), np.mean)
        train_mask_mat = scipy.io.loadmat("data/Danubio/changes_clipped_v3_masked_2percent.mat")
        train_mask = np.array(train_mask_mat["train_mask"], dtype=np.uint8)
        train_mask = np.ones(train_mask.shape,dtype=np.uint8)

        mask = np.zeros(shape=[t1.shape[0], t1.shape[1]], dtype='bool')
        mask[0,0] = True

        t1 = t1[:128,:128,:]
        t2 = t2[:128,:128,:]

    else:
        print("Wrong data set")
        exit()
    del mat
    time1 = time.time()
    cross_net = cross_network(t1, t2, mask, train_mask)
    cross_net.train_model()
    return cross_net.evaluation, time.time() - time1


if __name__ == "__main__":
    evaluated, times = run_model()
    tf.compat.v1.reset_default_graph()
    # os.system("rm -rf logs/train/*")
    # time.sleep(2)
    print(*evaluated, times, sep=", ")
