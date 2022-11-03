import SimpleITK as sitk
import numpy as np
import torch
from pathlib import Path
import monai_unet
from uNet_baseline.monai_unet import Net
import os
import shutil
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Lambda
from tensorflow.keras import backend as K

from torch.utils.data import DataLoader

from skimage import io
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import joblib
import cc3d
import csv
import sys


class Unet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.project_path = os.path.dirname(os.path.dirname(__file__)) + '/autoPET'
        # self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.input_path = self.project_path + '/resource'
        # self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.output_path = self.project_path
        # self.nii_path = '/opt/algorithm/'  # where to store the nii files
        self.nii_path = self.project_path
        # self.ckpt_path = '/opt/algorithm/epoch=777-step=64573.ckpt'
        self.ckpt_path = self.project_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        """
        function converts mha to nii format
        :param mha_input_path
        :param nii_out_path
        """
        img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        """
        function converts nii to mha format
        :param nii_input_path
        :param mha_out_path
        """
        img = sitk.ReadImage(nii_input_path)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        checks GPU availability
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, 'images/ct/'))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, 'images/pet/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'SUV.nii.gz'))
        self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'CTres.nii.gz'))
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(os.path.join(self.output_path, "PRED.nii.gz"),
                                os.path.join(self.output_path, uuid + ".mha"))
        print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self, inputs):
        """
        Your algorithm goes here
        """
        pass
        # return outputs

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        monai_unet.run_inference(self.ckpt_path, self.nii_path, self.output_path)
        print('Start output writing')
        self.write_outputs(uuid)


class History_Tensor(tf.keras.callbacks.Callback):
    # callback class and function adjustment for calculating test accuracy and loss

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        """
        function tracks on epochs
        :param epoch
        :param logs
        """
        loss, acc = self.model.evaluate(self.test_data)
        if 'test_loss' not in logs:
          logs['test_loss'] = []
        if 'test_accuracy' not in logs:
          logs['test_accuracy'] = []
        logs['test_loss'].append(loss)
        logs['test_accuracy'] = acc


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)  # normalizes feature vectors
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class Contrastive:

    def __init__(self):
        self.p_project = os.path.dirname(os.path.dirname(__file__)) + '/autoPET'
        self.p_resource = self.p_project + '/resource'
        # self.p_resource = '/sise/liorrk-group/edoli/PycharmProjects/autoPET/resource/nifti'

    def init_results(self):
        """
        function inits output file
        """
        l_cols = ['Model', 'Accuracy', 'Loss']
        # l_cols = ['Model', 'Accuracy', 'Loss', 'AUC', 'Recall', 'Precision', 'F1', 'PRAUC']
        df_results = pd.DataFrame(columns=l_cols)
        s_filename = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        p_output = self.p_project + '/' + s_filename + '.csv'
        return df_results, p_output

    def init(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print(
                '\n\nThis error most likely means that this notebook is not '
                'configured to use a GPU.  Change this in Notebook Settings via the '
                'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
            raise SystemError('GPU device not found')

        with tf.device('/device:GPU:0'):
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            random_image_gpu = tf.random.normal((100, 100, 100, 3))
            net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
            return tf.math.reduce_sum(net_gpu)

    def read_data(self, p_curr):
        """
        function reads data
        :param p_curr current path
        """
        CTres_Path = os.path.join(p_curr, "CTres.nii.gz")
        imgCTres = sitk.ReadImage(CTres_Path)
        imgCTres = np.expand_dims(sitk.GetArrayFromImage(imgCTres), 0)

        SUV_Path = os.path.join(p_curr, "SUV.nii.gz")
        imgSUV = sitk.ReadImage(SUV_Path)
        imgSUV = np.expand_dims(sitk.GetArrayFromImage(imgSUV), 0)
        return np.concatenate((imgCTres, imgSUV), 0)

    def to_uint8(self, data):
        """
        function converts format to uint8
        :param data
        """
        data -= data.min()
        data /= data.max()
        data *= 255
        return data.astype(np.uint8)

    def nii_to_jpgs(self, input_path, output_dir, rgb=False):
        """
        function converts format nii to jpg
        :param input_path
        :param output_dir
        :param rgb
        """
        output_dir = Path(output_dir)
        data = nib.load(input_path).get_fdata()
        *_, num_slices, num_channels = data.shape
        for channel in range(num_channels):
            volume = data[..., channel]
            volume = self.to_uint8(volume)
            channel_dir = output_dir / f'channel_{channel}'
            channel_dir.mkdir(exist_ok=True, parents=True)
            for slice in range(num_slices):
                slice_data = volume[..., slice]
                if rgb:
                    slice_data = np.stack(3 * [slice_data], axis=2)
                output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
                io.imsave(output_path, slice_data)

    def init_model(self, height, width, dim, num_classes=2):
        """
        function inits models
        :param height
        :param width
        :param dim
        :param num_classes
        """
        input_shape = (height, width, dim)

        # curr_model = tf.keras.applications.InceptionV3(include_top=False,
        #                                                weights=None,
        #                                                input_shape=input_shape,
        #                                                pooling="avg",
        #                                                classes=num_classes,
        #                                                classifier_activation="softmax",
        #                                                )

        # curr_model = tf.keras.applications.ResNet50V2(include_top=False,
        #                                               weights=None,
        #                                               input_shape=input_shape,
        #                                               pooling="avg"
        #                                               )

        curr_model = tf.keras.applications.Xception(include_top=False,
                                                    weights=None,
                                                    input_shape=input_shape,
                                                    pooling="avg",
                                                    classes=num_classes,
                                                    classifier_activation="softmax"
                                                    )

        # curr_model = self.unet(n_classes=num_classes, IMG_HEIGHT=height, IMG_WIDTH=width, IMG_CHANNELS=dim)

        return curr_model

    def unet(self, n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS=3):
        """
        function inits U-Net model
        :param n_classes
        :param IMG_HEIGHT
        :param IMG_WIDTH
        :param IMG_CHANNELS
        """
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # s = Lambda(lambda x: x / 255)(inputs)  # normalizes
        s = inputs

        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def nii2numpy(self, nii_path):
        """
        function converts nii to numpy
        :param nii_path
        """
        # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
        mask_nii = nib.load(str(nii_path))
        mask = mask_nii.get_fdata()
        pixdim = mask_nii.header['pixdim']
        voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
        return mask, voxel_vol

    def con_comp(self, seg_array):
        """
        function input a binary segmentation array output: an array with separated (indexed) connected components of the segmentation array
        :param seg_array
        """
        connectivity = 18
        conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
        return conn_comp

    def false_pos_pix(self, gt_array, pred_array):
        """
        function computes number of voxels of false positive connected components in prediction mask
        :param gt_array
        :param pred_array
        """
        pred_conn_comp = self.con_comp(pred_array)
        false_pos = 0
        for idx in range(1, pred_conn_comp.max() + 1):
            comp_mask = np.isin(pred_conn_comp, idx)
            if (comp_mask * gt_array).sum() == 0:
                false_pos = false_pos + comp_mask.sum()
        return false_pos

    def false_neg_pix(self, gt_array, pred_array):
        """
        function computes number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
        :param gt_array
        :param pred_array
        """
        gt_conn_comp = self.con_comp(gt_array)
        false_neg = 0
        for idx in range(1, gt_conn_comp.max() + 1):
            comp_mask = np.isin(gt_conn_comp, idx)
            if (comp_mask * pred_array).sum() == 0:
                false_neg = false_neg + comp_mask.sum()
        return false_neg

    def dice_score(self, mask1, mask2):
        """
        function computes foreground Dice coefficient
        :param mask1
        :param mask2
        """
        overlap = (mask1 * mask2).sum()
        sum = mask1.sum() + mask2.sum()
        dice_score = 2 * overlap / sum
        return dice_score

    def compute_metrics(self, nii_gt_path, nii_pred_path):
        """
        function computes evaluation scores
        :param nii_gt_path
        :param nii_pred_path
        """
        gt_array, voxel_vol = self.nii2numpy(nii_gt_path)
        pred_array, voxel_vol = self.nii2numpy(nii_pred_path)

        false_neg_vol = self.false_neg_pix(gt_array, pred_array) * voxel_vol
        false_pos_vol = self.false_pos_pix(gt_array, pred_array) * voxel_vol
        dice_sc = self.dice_score(gt_array, pred_array)

        return dice_sc, false_pos_vol, false_neg_vol

    def get_augmentation(self, x_train):
        """
        function sets augmentations
        :param x_train
        """
        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2),
        #     ]
        # )

        data_augmentation = keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.Normalization(),
                tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.02),
                tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
                tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
            ]
        )

        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomCrop(128, 128),
        #         layers.RandomZoom(0.5, 0.2),
        #         layers.RandomContrast(0.2),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2)
        #     ]
        # )

        data_augmentation.layers[0].adapt(x_train)  # sets the state of the normalization layer

        return data_augmentation

    def create_encoder(self, x_train, height, width, dim):
        """
        function inits encoder model
        :param x_train
        :param height
        :param width
        :param dim
        """
        shape = (height, width, dim)
        curr_model = self.init_model(height, width, dim)
        data_augmentation = self.get_augmentation(x_train)
        inputs = keras.Input(shape=shape)
        augmented = data_augmentation(inputs)
        outputs = curr_model(augmented)
        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-encoder")
        return model

    def add_projection_head(self, encoder, height, width, dim):
        """
        function inits projection model
        :param encoder
        :param height
        :param width
        :param dim
        """
        shape = (height, width, dim)
        projection_units = 128
        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        outputs = layers.Dense(projection_units, activation="relu")(features)
        model = keras.Model(
            inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
        )
        return model

    def create_classifier(self, encoder, height, width, dim, trainable=True):
        """
        function inits classifier
        :param encoder
        :param height
        :param width
        :param dim
        :param trainable
        """
        shape = (height, width, dim)
        learning_rate = 0.001
        hidden_units = 512
        num_classes = 2
        dropout_rate = 0.5

        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        features = layers.Dropout(dropout_rate)(features)
        features = layers.Dense(hidden_units, activation="relu")(features)
        features = layers.Dropout(dropout_rate)(features)
        outputs = layers.Dense(num_classes, activation="softmax")(features)

        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-clf")

        # _metrics = [keras.metrics.SparseCategoricalAccuracy(),
        #             keras.metrics.Recall(),
        #             keras.metrics.Precision(),
        #             tfa.metrics.F1Score(num_classes=num_classes, average='weighted'),
        #             keras.metrics.AUC()
        #             ]

        _metrics = [keras.metrics.SparseCategoricalAccuracy()]

        # _loss = 'binary_crossentropy'

        _loss = keras.losses.SparseCategoricalCrossentropy()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=_loss,
            metrics=_metrics
        )

        return model

    def plot_acc_loss(self, history, m_name, b_contrastive):
        """
        function plots model results
        :param history
        :param m_name model name
        :param b_contrastive flag for contrastive learning
        """
        if b_contrastive:
            plt.plot(history.history['sparse_categorical_accuracy'])
            # plt.plot(history.history['val_sparse_categorical_accuracy'])
        else:
            plt.plot(history.history['accuracy'])
            # plt.plot(history.history['val_accuracy'])
        # plt.plot(history.history['test_accuracy'])

        # accuracy plot
        plt.title('Model ' + m_name + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_acc = self.p_resource + '/' + m_name + '_acc.png'
        plt.savefig(p_save_acc)
        plt.show()
        plt.clf()

        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.plot(history.history['test_loss'])

        # loss plot
        plt.title('Model ' + m_name + ' Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_loss = self.p_resource + '/' + m_name + '_loss.png'
        plt.savefig(p_save_loss)
        plt.show()
        plt.clf()

    def save_model(self, o_model, s_model):
        """
        function saves model
        :param o_model model object
        :param s_model model name
        """
        print(f'Saved {s_model}.')
        p_output_model = self.p_project + '/contrastive.pkl'
        joblib.dump(o_model, p_output_model)

    def set_file_list(self):
        """
        function sets files in a list
        """
        l_pet, l_ct, l_ct_res, l_seg, l_suv = list(), list(), list(), list(), list()
        for root, dirs, files in os.walk(self.p_resource):
            for dir in dirs:
                child_root = root + '/' + dir
                for sub_root, sub_dirs, sub_files in os.walk(child_root):
                    for child_dir in sub_dirs:
                        grand_root = child_root + '/' + child_dir
                        for sub_root_child, sub_dirs_child, sub_files_child in os.walk(grand_root):
                            for curr_file in sub_files_child:
                                curr_file_path = os.path.join(sub_root_child, curr_file)
                                if 'PET' in curr_file:
                                    l_pet.append(curr_file_path)
                                elif 'CT' in curr_file and 'res' not in curr_file:
                                    l_ct.append(curr_file_path)
                                elif 'CT' in curr_file and 'res' in curr_file:
                                    l_ct_res.append(curr_file_path)
                                if 'SEG' in curr_file:
                                    l_seg.append(curr_file_path)
                                if 'SUV' in curr_file:
                                    l_suv.append(curr_file_path)
        return {'pet': l_pet, 'ct': l_ct, 'ct_res': l_ct_res, 'seg': l_seg, 'suv': l_suv}

    def process(self):
        """
        function preprocess
        """
        d_datasets = self.set_file_list()

        # height, width = 64, 64
        height, width = 128, 128
        # height, width = 256, 256
        # height, width = 512, 512
        # dim = 3
        dim = 1

        l_pet, l_ct = d_datasets['suv'], d_datasets['ct_res']
        pet, ct = np.array(l_pet), np.array(l_ct)
        x_pet, x_ct = list(), list()

        l_seg = d_datasets['seg']
        seg = np.array(l_seg)
        y, y_pet, y_ct = list(), list(), list()

        # length_x = 10
        length_x = len(pet)

        for i in tqdm(range(0, length_x)):
            curr_seg = sitk.ReadImage(seg[i])
            arr_seg = sitk.GetArrayFromImage(curr_seg)
            curr_value = np.expand_dims(arr_seg, 0)
            curr_class = int(curr_value[0][0][0][0])

            # seg = nib.load(seg[i])
            # nii_data = seg.get_fdata()
            # nii_aff = seg.affine
            # nii_hdr = seg.header
            # curr_class = int(nii_data[0][0][0])

            y.append(curr_class)

            curr_pet = nib.load(pet[i]).get_fdata()
            curr_pet = resize(curr_pet, (height, width))

            for i_slice in range(curr_pet.shape[2]):
                curr_img = curr_pet[:, :, i_slice]
                x_pet.append(curr_img)
                # if i_slice < 3:
                #     plt.imshow(curr_img)
                #     plt.show()
                y_pet.append(curr_class)

            curr_ct = nib.load(ct[i]).get_fdata()
            curr_ct = resize(curr_ct, (height, width))

            for i_slice in range(curr_ct.shape[2]):
                curr_img = curr_ct[:, :, i_slice]
                x_ct.append(curr_img)
                # if i_slice < 3:
                #     plt.imshow(curr_img)
                #     plt.show()
                y_ct.append(curr_class)

        x_pet, x_ct = np.array(x_pet), np.array(x_ct)
        count = np.where(y == 0)[0].size
        print(f'y count: {count}')
        y_pet, y_ct, y = np.array(y_pet), np.array(y_ct), np.array(y)
        return {'pet': x_pet, 'ct': x_ct}, {'pet': y_pet, 'ct': y_ct, 'all': y}

    def plot_image(self, x, y):
        """
        function plots images
        :param x model
        :param y model
        """
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))  # plots examples
        for i in tqdm(range(0, 5)):
            rand = np.random.randint(len(x))
            ax[i].imshow(x[rand])
            ax[i].axis('off')
            a = y[rand]
            if a == 1:
                ax[i].set_title('Diseased')
            else:
                ax[i].set_title('Non_Diseased')

    def set_callbacks(self, b_contrastive, test=None):
        """
        function sets callbacks
        :param b_contrastive
        :param test
        """
        _epoch_limit = 10

        if b_contrastive:
            _monitor = 'val_sparse_categorical_accuracy'
        else:
            _monitor = 'val_accuracy'

        early_stopping = EarlyStopping(monitor=_monitor,
                                       mode='max',
                                       patience=_epoch_limit,
                                       verbose=1)

        learning_rate = ReduceLROnPlateau(monitor=_monitor,
                                          mode='max',
                                          patience=5,
                                          factor=0.3,
                                          min_delta=0.00001)

        # board = tf.keras.callbacks.TensorBoard(self.p_project, update_freq=1)

        # history = History_Tensor(test)

        l_callbacks = [early_stopping]

        return l_callbacks

    def predict(self, d_x, d_y):
        """
        function predict
        :param d_x
        :param d_y
        """
        x_pet, x_ct = d_x['pet'], d_x['ct']
        y_pet, y_ct, y = d_y['pet'], d_y['ct'], d_y['all']

        df_results, p_output_results = self.init_results()

        # height, width = 64, 64
        height, width = 128, 128
        # height, width = 256, 256
        # height, width = 512, 512
        # dim = 3
        dim = 1

        # learning_rate = 0.01
        learning_rate = 0.001

        temperature = 0.05
        # temperature = 0.1
        # temperature = 0.2

        # num_epochs = 50
        num_epochs = 200
        # num_epochs = 1

        # batch_size = 32
        batch_size = 64
        # batch_size = 512
        # batch_size = 256
        # batch_size = 128

        b_contrastive = True
        # b_contrastive = False

        self.init()

        # y = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2)
        # y = y.transpose(1, -1).squeeze(-1)

        # x_pet, x_ct = shuffle(x_pet, random_state=42), shuffle(x_ct, random_state=42)

        # x_pet, x_ct = torch.tensor(x_pet), torch.tensor(x_ct)

        x_pet = np.reshape(x_pet, (x_pet.shape[0], height, width, dim))

        x_train, x_test, y_train, y_test = train_test_split(x_pet, y_pet, test_size=0.25, random_state=42)

        i_split = int(len(x_pet) * 0.9)
        train_images, val_images = x_pet[:i_split], x_pet[i_split:]
        train_set = DataLoader(train_images, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
        valid_set = DataLoader(val_images, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True, drop_last=False)

        l_callbacks = self.set_callbacks(b_contrastive, x_test)

        encoder = self.create_encoder(x_train, height, width, dim)

        encoder.summary()

        print('Training contrastive encoder...')

        encoder = self.create_encoder(x_train, height, width, dim)
        encoder_with_projection_head = self.add_projection_head(encoder, height, width, dim)
        encoder_with_projection_head.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=SupervisedContrastiveLoss(temperature),
        )
        encoder_with_projection_head.summary()

        history = encoder_with_projection_head.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

        print('Training projection network...')
        classifier = self.create_classifier(encoder, height, width, dim, trainable=False)
        s_model = 'Contrastive Model'

        # (1) without validation set
        history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, callbacks=l_callbacks)

        # (2) with validation set
        # curr_steps_per_epoch = x_train.shape[0] // batch_size
        # curr_validation_steps = x_val.shape[0] // batch_size
        # history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
        #                          validation_data=(x_val, y_val), callbacks=l_callbacks,
        #                          steps_per_epoch=curr_steps_per_epoch, validation_steps=curr_validation_steps,
        #                          shuffle=True)

        scores = classifier.evaluate(x_test, y_test)
        loss = round(scores[0] * 100, 3)
        accuracy = round(scores[1] * 100, 3)
        self.plot_acc_loss(history, s_model, b_contrastive)
        print(f'Model: {s_model}, Test Accuracy: {accuracy}%, Test Loss: {loss}%')

        df_results.loc[0, 'Model'] = s_model
        df_results.loc[0, 'Accuracy'] = accuracy
        df_results.loc[0, 'Loss'] = loss
        df_results.to_csv(path_or_buf=p_output_results, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')

        y_preds = classifier.predict(x_test).reshape(-1, 1)
        # y_probs = classifier.predict_proba(x_test)[:, 1]

        nii_pred_path = self.p_project+'/preds.csv'
        df_preds = pd.DataFrame.from_records(y_preds)
        df_preds.to_csv(path_or_buf=nii_pred_path, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
        p_preds = self.p_project + '/test/expected_output_uNet/PRED.nii.gz'
        nii_preds_file = nib.Nifti1Image(y_preds, np.eye(4))
        nib.save(nii_preds_file, p_preds)

        nii_gt_path = self.p_project+'/gt.csv'
        df_gt = pd.DataFrame.from_records(y_test)
        df_gt.to_csv(path_or_buf=nii_gt_path, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
        p_gt = self.p_project + '/test/expected_output_uNet/GT.nii.gz'
        nii_gt_file = nib.Nifti1Image(y_test, np.eye(4))
        nib.save(nii_gt_file, p_gt)

        # self.save_model(classifier, s_model)
        # classifier.save_weights(self.p_project + '/contrastive.h5')

        dice_sc, false_pos_vol, false_neg_vol = self.compute_metrics(nii_gt_path, nii_pred_path)
        csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
        csv_rows = ['y_true', dice_sc, false_pos_vol, false_neg_vol]
        with open('metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(csv_header)
            writer.writerows(csv_rows)


if __name__ == "__main__":
    # MAIN FLOW #
    con = Contrastive()
    d_x, d_y = con.process()
    con.predict(d_x, d_y)
