import os
from glob import glob

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import get_transforms


class HAM10000Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df["path"][idx]
        image = Image.open(image_path)
        label = self.df["target"][idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label


def get_datasets(metadata_path, num_users):
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
        for x in glob(os.path.join("data", "*", "*.jpg"))
    }

    df["path"] = df["image_id"].map(imageid_path.get)
    df["cell_type"] = df["dx"].map(lesion_type.get)
    df["target"] = pd.Categorical(df["cell_type"]).codes

    train, test = train_test_split(df, test_size=0.2)

    train = train.reset_index()
    test = test.reset_index()

    train_transforms, test_transforms = get_transforms()

    dataset_train = HAM10000Dataset(train, transform=train_transforms)
    dataset_test = HAM10000Dataset(test, transform=test_transforms)

    return dataset_train, dataset_test
