import os
from glob import glob

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.misc import get_transforms


class HAM10000Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df["path"][idx]
        image = Image.open(image_path).resize((64, 64))
        label = self.df["target"][idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label


class Shard(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(metadata_path):
    df = pd.read_csv(metadata_path)
    lesion_type = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions ",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }
    imageid_path = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(os.path.dirname(metadata_path), "images", "*.jpg"))
    }

    df["path"] = df["image_id"].map(imageid_path.get)
    df["cell_type"] = df["dx"].map(lesion_type.get)
    df["target"] = pd.Categorical(df["cell_type"]).codes

    train, test = train_test_split(df, test_size=0.2)

    train = train.reset_index()
    test = test.reset_index()

    train_transforms, test_transforms = get_transforms()

    dataset_train = HAM10000Dataset(train, transforms=train_transforms)
    dataset_test = HAM10000Dataset(test, transforms=test_transforms)

    return dataset_train, dataset_test
