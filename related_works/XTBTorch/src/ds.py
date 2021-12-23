import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from gen_utils import *
import pandas as pd
from torch.utils.data import random_split


class XrayDset(Dataset):
    def __init__(self, root_dir, tfm=transforms.ToTensor()):
        self.root_dir = root_dir
        self.fnames = os.listdir(root_dir)
        self.labels = [get_lbl_from_name(f) for f in self.fnames]
        self.tfm = tfm
        
    def __getitem__(self, index):
        I0 = Image.open(self.root_dir+self.fnames[index])
        I = self.tfm(I0)
        return I, self.labels[index]
    
    def __len__(self):
        return len(self.fnames)


def get_train_test_data(dataset_dir, type='all'):
    eval_ratio = 1/8

    train_file = f'{dataset_dir}/experiments/train.csv'
    test_file = f'{dataset_dir}/experiments/test.csv'

    df_train = pd.read_csv(train_file, index_col='image')
    if type == 'H':
        df_train = df_train[df_train['type']=='H']
    elif type == 'D':
        df_train = df_train[df_train['type']=='D']
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
    if type == 'H':
        df_test = df_test[df_test['type']=='H']
    elif type == 'D':
        df_test = df_test[df_test['type']=='D']
        
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


class TBDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset # dataset_dir:'{image_folder}/{image}_{id}.jpg
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        id = int(self.dataset[index][1])
        img = Image.open(img_path).convert('L')
        img = img.resize((128, 128))
        # img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, id