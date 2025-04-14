import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPet(Dataset):
    def __init__(self, transform=None):
        self.classes = [
            "Abyssinian",
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "Bengal",
            "Birman",
            "Bombay",
            "boxer",
            "British_Shorthair",
            "chihuahua",
            "Egyptian_Mau",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "Maine_Coon",
            "miniature_pinscher",
            "newfoundland",
            "Persian",
            "pomeranian",
            "pug",
            "Ragdoll",
            "Russian_Blue",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "Siamese",
            "Sphynx",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        column_name = ["Image", "ID", "SPECIES", "BREED ID"]
        self.label_df = pd.read_csv(
            "../annotations/list.txt",
            sep="\s+",
            skiprows=6,
            header=None,
            names=column_name,
        )

        self.ids = []

        self.images_folder = "../images/"
        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(".jpg"):
                if not self.label_df[self.label_df["Image"] == filename[:-4]].empty:
                    self.ids.append(filename[:-4])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        image_path = self.images_folder + self.ids[item] + ".jpg"
        image = self.transform(Image.open(image_path).convert("RGB"))
        label = (
            self.label_df[self.label_df["Image"] == self.ids[item]]["ID"].iloc[0] - 1
        )

        return image, label, self.ids[item]
