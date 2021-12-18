import os
import pandas as pd
import numpy as np

img_dir = '/media/data/mu/ML2/data2/external_data/'
exp_dir = f'{img_dir}/experiments'
all_img_path = f'{exp_dir}/dataset.csv'
train_data_path = f'{exp_dir}/train.csv'
test_data_path = f'{exp_dir}/test.csv'

all_imgs_list = []
all_imgs_ids_list = []

train_imgs_list = []
train_ids_list = []

test_imgs_list = []
test_ids_list = []

train_test_split_ratio = 0.8

# folders = ['Diabetes', 'HIV', 'TB-only']
folders = os.listdir(img_dir)

for folder in folders:
    if folder == 'experiments':
        continue

    _img_path = f'{img_dir}/{folder}'
    types = os.listdir(_img_path)

    for type in types:  ### TB or normal

        if 'Normal' == type:
            id = 0
        elif 'Tuberculosis' == type:
            id = 1
        else:
            continue

        img_path = f'{img_dir}/{folder}/{type}'
        filename_list = os.listdir(img_path)
        filename_list = [f'{folder}/{type}/{file}' for file in filename_list]

        print('===id===', id)
        num_files = len(filename_list)
        all_imgs_list.extend(filename_list)
        all_imgs_ids_list.extend([id]*num_files)

        print('all', num_files)
        num_train = int(num_files * train_test_split_ratio)
        print('numtrain-------', num_train)
        train_sample_index = np.random.choice(num_files, size= num_train, replace=False)
        train_imgs = np.array(filename_list)[train_sample_index]
        train_imgs_list.extend(train_imgs)
        train_ids_list.extend([id]*num_train)
        print('train', len(train_sample_index), train_sample_index)

        test_sample_index = list(set(range(num_files)) - set(train_sample_index))
        print('len test', len(test_sample_index))
        test_imgs = np.array(filename_list)[test_sample_index]
        test_imgs_list.extend(test_imgs)
        test_ids_list.extend([id]*(num_files-num_train))
        print('test', len(test_sample_index), test_sample_index)
        print('filenames', len(filename_list), id)

print('all_imgs_list', len(all_imgs_list), len(all_imgs_ids_list))
print('trainimages', len(train_imgs_list), len(train_ids_list))
print('testimages', len(test_imgs_list), len(test_ids_list))

columns = ['image', 'id']
df_all = pd.DataFrame(list(zip(all_imgs_list, all_imgs_ids_list)), columns=columns)
df_train = pd.DataFrame(list(zip(train_imgs_list, train_ids_list)), columns=columns)
df_test = pd.DataFrame(list(zip(test_imgs_list, test_ids_list)), columns=columns)

df_all.to_csv(f'{all_img_path}', index=False)
df_train.to_csv(f'{train_data_path}', index=False)
df_test.to_csv(f'{test_data_path}', index=False)





        
    

