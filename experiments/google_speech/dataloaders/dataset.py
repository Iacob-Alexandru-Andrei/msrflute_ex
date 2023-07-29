# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import numpy as np
from core.dataset import BaseDataset
from PIL import Image
import torch
from experiments.google_speech.dataloaders.preprocessing import SPEECH
import torchvision.transforms as transforms


from FedScale.fedscale.dataloaders.speech import BackgroundNoiseDataset
from FedScale.fedscale.dataloaders.transforms_stft import (
    AddBackgroundNoiseOnSTFT,
    DeleteSTFT,
    FixSTFTDimension,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
)
from FedScale.fedscale.dataloaders.transforms_wav import (
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    FixAudioLength,
    LoadAudio,
    ToMelSpectrogram,
    ToTensor,
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

        data = self.data[idx]
        target = int(self.labels[idx])

        data = {
            "path": data,
            "target": target,
        }

        if self.transform is not None:
            data = self.transform(data)

        return torch.unsqueeze(data["input"], 0), data["target"]

    def __len__(self):
        return len(self.data)

    def load_data(self, data, test_only, filter_less, filter_more, data_loc):
        """Wrapper method to read/instantiate the dataset"""
        dataset = data
        if isinstance(data, str):
            dataset = SPEECH(
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


def get_train_transform():
    global SPEECH, BackgroundNoiseDataset, AddBackgroundNoiseOnSTFT, DeleteSTFT, FixSTFTDimension, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, ToMelSpectrogramFromSTFT, ToSTFT, ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor

    data_loc = "/datasets/FedScale/openImg/train"

    bkg = "_background_noise_"
    data_aug_transform = transforms.Compose(
        [
            ChangeAmplitude(),
            ChangeSpeedAndPitchAudio(),
            FixAudioLength(),
            ToSTFT(),
            StretchAudioOnSTFT(),
            TimeshiftAudioOnSTFT(),
            FixSTFTDimension(),
        ]
    )
    bg_dataset = BackgroundNoiseDataset(
        os.path.join("/datasets/FedScale/google_speech/google_speech/", bkg),
        data_aug_transform,
    )
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = transforms.Compose(
        [
            ToMelSpectrogramFromSTFT(n_mels=32),
            DeleteSTFT(),
            ToTensor("mel_spectrogram", "input"),
        ]
    )

    transform = transforms.Compose(
        [
            LoadAudio(),
            data_aug_transform,
            add_bg_noise,
            train_feature_transform,
        ]
    )

    return transform


def get_test_transform():
    valid_feature_transform = transforms.Compose(
        [ToMelSpectrogram(n_mels=32), ToTensor("mel_spectrogram", "input")]
    )
    transform = transforms.Compose(
        [LoadAudio(), FixAudioLength(), valid_feature_transform]
    )
    return transform
