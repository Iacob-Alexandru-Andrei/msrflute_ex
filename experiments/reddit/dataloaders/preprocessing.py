import os
import json
import csv
from collections import defaultdict
from torchvision import datasets, transforms
from multiprocessing import Pool, cpu_count


from FedScale.fedscale.dataloaders.nlp import load_and_cache_examples


class REDDIT:
    def __init__(
        self, data_path: str, filter_less, filter_more, evaluate, args, tokenizer
    ):

        args["data_dir"] = args["data_loc"]
        feature_based_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=evaluate
        )

        train_dict = {
            "users": [],
            "num_samples": [],
            "user_data": defaultdict(list),
            "user_data_label": defaultdict(list),
        }
        feature_based_dataset.client_mapping = {
            str(k): v for k, v in feature_based_dataset.client_mapping.items()
        }
        train_dict["users"] = sorted(feature_based_dataset.client_mapping.keys())

        train_dict["user_data"] = {
            user_id: [feature_based_dataset.data[v] for v in vals]
            for user_id, vals in feature_based_dataset.client_mapping.items()
        }

        train_dict["user_data_label"] = {k: -1 for k in train_dict["user_data"].keys()}
        to_delete = []
        del_indicies = []
        for i, user_id in enumerate(train_dict["users"]):
            if (
                len(train_dict["user_data"][user_id]) >= filter_less
                and len(train_dict["user_data"][user_id]) <= filter_more
            ):
                train_dict["num_samples"].append(len(train_dict["user_data"][user_id]))
            else:
                to_delete.append(user_id)
                del_indicies.append(i)
        filter(
            train_dict=train_dict,
            to_delete=to_delete,
            del_indicies=del_indicies,
        )
        self.data = train_dict


def filter(train_dict: dict, to_delete: list, del_indicies=list):
    for user_id in to_delete:
        del train_dict["user_data"][user_id]
        del train_dict["user_data_label"][user_id]
    train_dict["users"] = [
        id for i, id in enumerate(train_dict["users"]) if i not in del_indicies
    ]
