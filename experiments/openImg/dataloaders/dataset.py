# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from core.dataset import BaseDataset
from PIL import Image
from experiments.openImg.dataloaders.preprocessing import OPENIMG
import torchvision.transforms as transforms

train_transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(224),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop((128,128)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class Dataset(BaseDataset):
    def __init__(self, data, test_only=False, user_idx=0, **kwargs):

        self.test_only = test_only
        self.user_idx = user_idx
        args = kwargs["args"]
        if data is None:
            (
                self.user_list,
                self.user_data,
                self.user_data_label,
                self.num_samples,
            ) = (
                [],
                {},
                {},
                [],
            )
        else:
            # Get all data
            (
                self.user_list,
                self.user_data,
                self.user_data_label,
                self.num_samples,
            ) = self.load_data(
                data=data,
                test_only=(test_only or user_idx == -1),
                filter_less=args["filter_less"],
                filter_more=args["filter_more"],
                data_loc=args["data_loc"],
            )

            if user_idx == -1:
                self.user = "test_only"
                self.data = self.user_data.values() if self.user_data else []
                self.labels = (
                    self.user_data_label.values() if self.user_data_label else []
                )
                self.transform = test_transform
            else:
                if self.test_only:  # combine all data into single array
                    self.user = "test_only"
                    self.data = self.user_data.values() if self.user_data else []
                    self.labels = (
                        self.user_data_label.values() if self.user_data_label else []
                    )
                    self.transform = test_transform
                else:  # get a single user's data
                    if user_idx is None:
                        raise ValueError("in train mode, user_idx must be specified")

                    self.user = self.user_list[user_idx]
                    self.data = self.user_data[self.user]
                    self.labels = self.user_data_label[self.user]
                    self.transform = train_transform

    def __getitem__(self, idx):

        img = Image.open(self.data[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.transform(img)

        target = int(self.labels[idx])

        return img, target

    def __len__(self):
        return len(self.data)

    def load_data(self, data, test_only, filter_less, filter_more, data_loc):
        """Wrapper method to read/instantiate the dataset"""
        dataset = data
        if isinstance(data, str):
            dataset = OPENIMG(
                data,
                filter_less=filter_less,
                filter_more=filter_more,
                data_loc=data_loc,
            )
            dataset = dataset.data

        users = dataset["users"]
        data = dataset["user_data"]
        labels = dataset["user_data_label"]
        num_samples = dataset["num_samples"]

        return users, data, labels, num_samples
