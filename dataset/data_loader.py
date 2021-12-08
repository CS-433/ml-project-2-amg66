from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import TBDataset
import glob

def split_train_test(dataset_dir):
    data_list = glob.glob(f'{dataset_dir}/*')

    # TODO
    train = data_list[:10]
    test = data_list[10:20]
    num_cls = 2

    return train, test, num_cls


def make_dataloader(dataset_dir, batch_size=64):

    # TODO
    train_transforms = False
    test_transforms = False

    train_data, test_data, _ = split_train_test(dataset_dir)

    train_set = TBDataset(train_data, transform=train_transforms)

    # TODO
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
    )

    test_set = TBDataset(test_data, transform=test_transforms)

    # TODO
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
    )

    return train_loader, test_loader


