import os
import json
import csv
from collections import defaultdict

data_loc = "/datasets/FedScale/openImg/train"


class OPENIMG:
    def __init__(self, data_path: str, filter_less, filter_more, data_loc):

        train_dict = {
            "users": [],
            "num_samples": [],
            "user_data": defaultdict(list),
            "user_data_label": defaultdict(list),
        }
        json_path = "./temp/speech_json_partition.json"
        if "json" in data_path:
            with open(json_path, mode="r") as js:
                train_dict = json.load(js)
        elif os.path.exists(data_path):
            if data_path.endswith(".csv"):
                with open(data_path, mode="r") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    next(csv_reader)
                    for row in csv_reader:
                        client_id, sample_path, _, label_id = row
                        if client_id not in train_dict["users"]:
                            train_dict["users"].append(client_id)
                        train_dict["user_data"][client_id].append(
                            os.path.join(data_loc, sample_path)
                        )
                        train_dict["user_data_label"][client_id].append(label_id)
                to_delete = []
                del_indicies = []
                for i, user_id in enumerate(train_dict["users"]):
                    if (
                        len(train_dict["user_data"][user_id]) >= filter_less
                        and len(train_dict["user_data"][user_id]) <= filter_more
                    ):
                        train_dict["num_samples"].append(
                            len(train_dict["user_data"][user_id])
                        )
                    else:
                        to_delete.append(user_id)
                        del_indicies.append(i)
                filter(
                    train_dict=train_dict,
                    to_delete=to_delete,
                    del_indicies=del_indicies,
                )

                with open(json_path, mode="w") as out_file:
                    json.dump(train_dict, out_file)
                print(" Dictionaries ready .. ")
        else:
            print("Dataset does not exist")
        self.data = train_dict


def filter(train_dict: dict, to_delete: list, del_indicies=list):
    for user_id in to_delete:
        del train_dict["user_data"][user_id]
        del train_dict["user_data_label"][user_id]
    train_dict["users"] = [
        id for i, id in enumerate(train_dict["users"]) if i not in del_indicies
    ]
