# coding:utf-8
import cv2
import pandas as pd
import numpy as np
import params
import random
import h5py
import os
from keras.utils import HDF5Matrix


def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    ## Chop off 1/3 from the top and cut bottom 50px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0]/3):shape[0]-50, 0:shape[1]]
    img = cv2.resize(img, (params.FLAGS.img_w, params.FLAGS.img_h), interpolation=cv2.INTER_AREA)
    return img

def img_change_brightness(img):
    """ Changing brightness of img to simulate day and night conditions
    :param img: The image to be processed
    :return: Returns the processed image
   """
    # Convert the image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute a random brightness value and apply to the image
    brightness = np.random.uniform(0.5,1.5)
    img[:, :, 2] = img[:, :, 2] * brightness
    # Convert back to RGB and return
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def img_horizontal_flip(img):
    img = img[:,::-1,:]
    return img

def create_h5data():
    h5_path = params.data_path
    if os.path.exists(h5_path):
        return

    f = h5py.File(h5_path, "w")
    origin_length = 24300
    train_origin_length = int(origin_length * 0.8)
    train_length = 2 * train_origin_length
    val_length = origin_length - train_origin_length
    test_length = 2700

    images = f.create_dataset('images', (origin_length, 720, 1280, params.FLAGS.img_c), dtype='uint8')
    labels = f.create_dataset('labels', (origin_length,), dtype='float32')
    tmp_images = f.create_dataset('tmp_images', (train_length,params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c), dtype='uint8')
    tmp_labels = f.create_dataset('tmp_labels', (train_length,), dtype='float32')
    train_images = f.create_dataset('train_images', (train_length, params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c), dtype='uint8')
    train_labels = f.create_dataset('train_labels', (train_length,), dtype='float32')
    val_images = f.create_dataset('val_images', (val_length, params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c), dtype='uint8')
    val_labels = f.create_dataset('val_labels', (val_length,), dtype='float32')
    test_images = f.create_dataset('test_images', (test_length, params.FLAGS.img_h, params.FLAGS.img_w, params.FLAGS.img_c), dtype='uint8')
    test_labels = f.create_dataset('test_labels', (test_length,), dtype='float32')

    x_train = []
    y_train = []

    # 读取所有原始数据
    image_index = 0
    lable_index = 0
    for i in range(1, 10):
        # 读取转向角度
        y_train_original = []
        pathY = './epochs/epoch0' + str(i) + '_steering.csv'
        wheel_sig = pd.read_csv(pathY)
        y_train_original.extend(wheel_sig['wheel'].values)
        length = len(y_train_original)
        labels[lable_index:lable_index + length] = y_train_original
        lable_index += len(y_train_original)

        # 读取图片
        pathX = './epochs/epoch0' + str(i) + '_front.mkv'
        cap = cv2.VideoCapture(pathX)

        while True:
            ret, img = cap.read()
            if (ret):
                images[image_index] = img
                image_index += 1
            else:
                break
        cap.release()

    # 生成校验数据
    index = [i for i in range(origin_length)]
    random.shuffle(index)
    for i in range(val_length):
        val_images[i] = img_pre_process(images[index[i]])
        val_labels[i] = labels[index[i]]

    # 生成临时数据
    for i in range(val_length, origin_length):
        img = img_pre_process(images[index[i]])
        tmp_images[i - val_length] = img
        tmp_labels[i - val_length] = labels[index[i]]

        flip_img = img_horizontal_flip(img)
        tmp_images[i - val_length + train_origin_length] = flip_img
        tmp_labels[i - val_length + train_origin_length] = -labels[index[i]]

    # 打乱临时数据生成训练数据
    index = [i for i in range(train_length)]
    random.shuffle(index)
    for i in range(train_length):
        train_images[i] = tmp_images[index[i]]
        train_labels[i] = tmp_labels[index[i]]

    del f['images']
    del f['labels']
    del f['tmp_images']
    del f['tmp_labels']

    # 生成测试集
    path10 = './epochs/epoch10_front.mkv'
    cap = cv2.VideoCapture(path10)
    index = 0
    while True:
        ret, img = cap.read()
        if (ret):
            test_images[index] = img_pre_process(img)
            index += 0
        else:
            break
    cap.release()

    y_test_label = []
    pathY = './epochs/epoch10_steering.csv'
    wheel_sig = pd.read_csv(pathY)
    y_test_label.extend(wheel_sig['wheel'].values)
    test_labels[:] = y_train_original

    f.close()

def remove_data():
    h5_path = './epochs/deep_tesla.hdf5'
    if os.path.exists(h5_path):
        os.remove(h5_path)

def load_data():
    create_h5data()
    data_file = params.data_path
    X_train = HDF5Matrix(data_file, 'train_images')
    y_train = HDF5Matrix(data_file,'train_labels')
    X_val = HDF5Matrix(data_file,'val_images')
    y_val = HDF5Matrix(data_file,'val_labels')
    X_test = HDF5Matrix(data_file,'test_images')
    y_test = HDF5Matrix(data_file,'test_labels')
    return (X_train, y_train),(X_val,y_val),(X_test, y_test)
