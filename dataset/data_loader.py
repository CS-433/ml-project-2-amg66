import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
torch.manual_seed(17)
from dataset.dataset import TBDataset, get_id_from_imname
import glob
import numpy as np
import pandas as pd

def get_random_item(dataset, test_ratio=0.2):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_set, test_set = dataset[train_indices], dataset[test_indices]

    return train_set, test_set

def get_train_test_data(dataset_dir, type=None):
    eval_ratio = 1/8

    train_file = f'{dataset_dir}/experiments/train.csv'
    test_file = f'{dataset_dir}/experiments/test.csv'

    df_train = pd.read_csv(train_file, index_col='image')
    train_val_imgs = df_train.index.tolist()
    train_val_imgs = [f'{dataset_dir}/{img}' for img in train_val_imgs]
    train_val_ids = df_train['id'].tolist()
    train_val_data = [train_val_imgs, train_val_ids]
    train_val_data = np.array(train_val_data).T

    val_size = int(eval_ratio * len(train_val_imgs))
    train_size = len(train_val_imgs) - val_size
    train_set, val_set = random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print('---------------------------------------------------------------')
    print('------- training samples:', train_size, 'validation samples:', val_size, ' -------')
    positive_num = len([img_id[1] for img_id in train_set if int(img_id[1])==1])
    negative_num = len([img_id[1] for img_id in train_set if int(img_id[1])==0])
    print('During training: pos. vs neg. ||', positive_num, 'vs', negative_num, '】')
    positive_num = len([img_id[1] for img_id in val_set if int(img_id[1])==1])
    negative_num = len([img_id[1] for img_id in val_set if int(img_id[1])==0])
    print('During validation: pos. vs neg. 【', positive_num, 'vs', negative_num, '】')

    df_test = pd.read_csv(test_file, index_col='image')
    test_imgs = df_test.index.tolist()
    test_imgs = [f'{dataset_dir}/{img}' for img in test_imgs]
    test_ids = df_test['id'].tolist()
    test_set = [test_imgs, test_ids]
    test_set = np.array(test_set).T

    positive_num = len([id for id in test_ids if id==1])
    negative_num = len([id for id in test_ids if id==0])
    print('During testing: pos. vs neg. 【', positive_num, 'vs', negative_num, '】')
    print('---------------------------------------------------------------')

    return train_set, val_set, test_set


def split_train_test(dataset_dir_list, train_ratio=0.8):

    data_list = []
    for dataset_dir in dataset_dir_list:
        data = glob.glob(f'{dataset_dir}/*')
        data_list.extend(data)

    id_list = [get_id_from_imname(f) for f in data_list]
    num_cls, counts = np.unique(id_list, return_counts=True)

    pos_list = [data for data, id in zip(data_list, id_list) if id ==1]
    neg_list = [data for data, id in zip(data_list, id_list) if id ==0]

    train_pos_dataset, test_pos_dataset = get_random_item(np.array(pos_list))
    train_neg_dataset, test_neg_dataset = get_random_item(np.array(neg_list))

    train = np.append(train_pos_dataset, train_neg_dataset)
    test = np.append(test_pos_dataset, test_neg_dataset)

    return train, test, num_cls


def create_loader(dataset_dir, 
                  batch_size=64, 
                  input_size = 224,
                  mean = [0.485, 0.456, 0.406],
                  std = [0.229, 0.224, 0.225],
                  num_workers=8):

    # TODO
    train_transforms = transforms.Compose([
                            transforms.Resize(int(1.1*input_size)),
                            transforms.RandomResizedCrop(input_size),
                            transforms.RandomVerticalFlip(),
                            transforms.CenterCrop(input_size),
                            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                            transforms.RandomRotation(degrees=(0, 180)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    test_transforms = transforms.Compose([
                            transforms.Resize(int(1.1*input_size)),
                            transforms.CenterCrop(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)          
                        ])

    train_data, val_data, test_data = get_train_test_data(dataset_dir)

    train_set = TBDataset(train_data, transform=train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True#, persistent_workers=True
    )

    val_set = TBDataset(val_data, transform=test_transforms)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True#, persistent_workers=True
    )

    test_set = TBDataset(test_data, transform=test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True#, persistent_workers=True
    )

    num_train_data = len(train_data)
    num_val_data = len(val_data)
    num_test_data = len(test_data)

    return train_loader, val_loader, test_loader, num_train_data, num_val_data, num_test_data