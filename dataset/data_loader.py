from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset.dataset import TBDataset, get_id_from_imname
import glob
import numpy as np


def get_random_item(dataset, test_ratio=0.2):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_set, test_set = dataset[train_indices], dataset[test_indices]

    return train_set, test_set


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


def create_loader(dataset_dir, batch_size=64):

    # TODO
    train_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])


    train_data, test_data, _ = split_train_test(dataset_dir)

    train_set = TBDataset(train_data, transform=train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
    )

    test_set = TBDataset(test_data, transform=test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
    )

    return train_loader, test_loader
