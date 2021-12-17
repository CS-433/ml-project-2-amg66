from torch.utils.data.dataset import Dataset
import os
from PIL import Image

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def get_id_from_imname(imname):
    imname = os.path.split(imname)[1]
    name = os.path.splitext(imname)[0]

    id = name[name.find('_')+1:]
    return int(id)

class TBDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset # dataset_dir:'{image_folder}/{image}_{id}.jpg
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        id = int(self.dataset[index][1])
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, id