# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import numpy as np
from core.dataset import BaseDataset
from PIL import Image
import torch
from experiments.reddit.dataloaders.preprocessing import REDDIT
import torchvision.transforms as transforms
from utils.utils import get_tokenizer


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
                args=args,
                tokenizer=get_tokenizer(args["model"]),
            )

            if user_idx == -1:
                self.user = "test_only"
                self.data = self.user_data.values() if self.user_data else []
                self.labels = (
                    self.user_data_label.values() if self.user_data_label else []
                )
                self.transform = get_test_transform()
            else:
                if self.test_only:  # combine all data into single array
                    self.user = "test_only"
                    self.data = self.user_data.values() if self.user_data else []
                    self.labels = (
                        self.user_data_label.values() if self.user_data_label else []
                    )
                    self.transform = get_test_transform()
                else:  # get a single user's data
                    if user_idx is None:
                        raise ValueError("in train mode, user_idx must be specified")

                    self.user = self.user_list[user_idx]
                    self.data = self.user_data[self.user]
                    self.labels = self.user_data_label[self.user]
                    self.transform = get_train_transform()

    def __getitem__(self, idx):

        cur_data = self.data[idx]

        if self.transform is not None:
            cur_data = self.transform(cur_data)
        return cur_data

    def __len__(self):
        return len(self.data)

    def load_data(
        self, data, test_only, filter_less, filter_more, args, tokenizer, **kwargs
    ):
        """Wrapper method to read/instantiate the dataset"""
        dataset = data
        if isinstance(data, str):
            dataset = REDDIT(
                data,
                filter_less=filter_less,
                filter_more=filter_more,
                args=args,
                evaluate=test_only,
                tokenizer=tokenizer,
            )
            dataset = dataset.data

        users = dataset["users"]
        data = dataset["user_data"]
        labels = dataset["user_data_label"]
        num_samples = dataset["num_samples"]

        return users, data, labels, num_samples


def get_train_transform():
    return lambda input: torch.tensor(input, dtype=torch.long)


def get_test_transform():
    return None
