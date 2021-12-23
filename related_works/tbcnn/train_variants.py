import os
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
import cv2
import torch
from torch.utils.data import random_split
import imageio
from PIL import Image

import gc

import train_loop

def shuffle(images, labels):
    """Return shuffled copies of the arrays, keeping the indexes of
    both arrays in corresponding places
    """

    cp_images = np.copy(images)
    cp_labels = np.copy(labels)

    rng_state = np.random.get_state()
    np.random.shuffle(cp_images)
    np.random.set_state(rng_state)
    np.random.shuffle(cp_labels)

    return cp_images, cp_labels
    
def split_train_and_test(images, labels, ratio=0.8):
    """Splits the array into two randomly chosen arrays of training and testing data.
    ratio indicates which percentage will be part of the training set."""

    images, labels = shuffle(images, labels)

    split = int(images.shape[0] * ratio)

    training_images = images[:split]
    training_labels = labels[:split]

    test_images = images[split:]
    test_labels = labels[split:]

    train_data = [training_images, training_labels]
    test_data =  [test_images, test_labels]

    return train_data, test_data

def get_img_by_path(path):
    print('path', path)
    img = Image.open(path).convert('L')
    print('img', img.size)
    # img = imageio.imread(path).convert('L')
    # img = cv2.resize(img, (512, 512))
    img_float = np.array(img).astype(np.float32)
    return img_float

def norm_images(image_list):
    images = np.stack(image_list)
    # Input normalization
    # Remove mean
    images -= np.mean(images)
    # Divide by standard deviation
    images /= np.std(images)

    # Add dummy channel layer
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))
    return images

def get_our_train_test_data(dataset_dir, type=None):
    # eval_ratio = 1/8


    if type != 'combine':
        train_file = f'{dataset_dir}/experiments/train.csv'
        test_file = f'{dataset_dir}/experiments/test.csv'
        df_train = pd.read_csv(train_file, index_col='image')
        df_test = pd.read_csv(test_file, index_col='image')

    else:
        data_file = f'{dataset_dir}/experiments/dataset.csv'
        all_data = pd.read_csv(data_file, index_col='image')
        df_test = all_data[all_data['type']=='D']
        df_train = all_data.drop(df_test.index.tolist())
        print('df_train', len(all_data), len(df_test), len(df_train))

    if type == 'H':
        df_train = df_train[df_train['type']=='H']
    elif type == 'D':
        df_train = df_train[df_train['type']=='D']
    train_val_imgs = df_train.index.tolist()
    train_val_imgs = [f'{dataset_dir}/{img}' for img in train_val_imgs]
    print('len', len(train_val_imgs))
    train_val_imgs = [get_img_by_path(path) for path in train_val_imgs]
    train_val_imgs = norm_images(train_val_imgs)
    train_val_ids = np.array(df_train['id'].tolist())
    train_val_data = [train_val_imgs, train_val_ids]

    if type == 'H':
        df_test = df_test[df_test['type']=='H']
    elif type == 'D':
        df_test = df_test[df_test['type']=='D']
    test_imgs = df_test.index.tolist()
    test_imgs = [f'{dataset_dir}/{img}' for img in test_imgs]
    test_imgs = [get_img_by_path(path) for path in test_imgs]
    test_imgs = norm_images(test_imgs)
    test_ids = np.array(df_test['id'].tolist())
    print(test_ids)
    test_set = [test_imgs, test_ids]

    return train_val_data, test_set
    

def create_sets(num, images, labels):
    """Splits the array into num equally sized sets."""

    images, labels = shuffle(images, labels)

    set_size = images.shape[0] // num
    remaining = images.shape[0] - set_size * num

    image_sets = []
    label_sets = []
    offset = 0
    for i in range(num):
        extra = 1 if i < remaining else 0

        image_sets.append(images[i*set_size + offset:i*set_size + set_size + offset + extra])
        label_sets.append(labels[i*set_size + offset:i*set_size + set_size + offset + extra])

        offset += extra
    
    return image_sets, label_sets

def get_rotations(num, image_sets, label_sets):
    """Create rotations of the training and test sets for cross validation training
    This means if image_sets = [A, B, C] the output will be [[A, B], [B, C], [A, C]]
    for the training set."""

    training_sets = []
    test_sets = []
    for i in range(num):
        test_sets.append((
            image_sets[i],
            label_sets[i]
        ))

        training_sets.append((
            np.concatenate([s for j, s in enumerate(image_sets) if j != i]),
            np.concatenate([s for j, s in enumerate(label_sets) if j != i])
        ))
    
    return training_sets, test_sets

def train_single(inFile, size=512, method=0, type='all'):
    """Train network a single time using the given files as input.

    inFile => path without extension (more than one file will be read)
    """

    print('Training...')

    # Load data
    import time 
    start = time.time()
    images = np.load(inFile + '.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    ori_training, ori_test = split_train_and_test(images, labels)
    t1 = time.time() 
    print('t1', t1-start)

    if method != 0:
        dataset_dir = '/media/data/mu/ML2/data2/our_data_processed'
        print('dataset', dataset_dir)
        my_training, my_test = get_our_train_test_data(dataset_dir, type=type)

    t2 = time.time()
    print('t2:', t2-t1)
    # print('ori train', ori_training[0].shape)
    # print('my train', my_training[0].shape)

    # Create training and test sets
    if method == 0:
        training, test = ori_training, ori_test
    elif method == 1:
        training, test = ori_training, my_test
    elif method == 2:
        training, test = my_training, my_test

    train_loop.train_net(training, test, size=size, out_name=f'result_m{method}.json')

def train_cross_validation(inFile, sets=3, size=512):
    """Train network multiple times in a cross validation fashon, in order to
    cover all the dataset in the tests and avoid the effect of outliers.

    inFile => path without extension (more than one file will be read)
    sets   => number of cross validation sets (training will be repeated this many times
              and the size of the test set will be dataset_size / sets)
    """

    print('Starting {}-fold cross validation study...'.format(sets))

    # Load data
    images = np.load(inFile + '.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    # Create training and test sets for the cross validation study
    image_sets, label_sets = create_sets(sets, images, labels)

    training_sets, test_sets = get_rotations(sets, image_sets, label_sets)
    # import matplotlib.pyplot as plt
    # plt.imshow(training_sets[0][0][0,:,:,0]); plt.show();

    for i in range(sets):
        print('Set {}'.format(i+1))

        train_loop.train_net(
            training_sets[i],
            test_sets[i],
            size=size,
            run_name='Set {} ({})'.format(i+1, datetime.now().strftime(r'%Y-%m-%d_%H:%M')),
        )

        tf.reset_default_graph()
        gc.collect()
