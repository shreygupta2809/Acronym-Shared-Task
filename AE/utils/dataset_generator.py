import pandas as pd
import json
import sys


class Dataset:
    def __init__(self, filename):
        self.filename = filename
        self.output_file = self.filename.split(".")[0] + ".csv"
        self.convert_to_df()
        self.write_to_csv()

    def convert_to_df(self):
        with open(self.filename) as file:
            data = json.load(file)
            if "test" not in self.filename:
                dataset = [
                    [
                        sample["id"],
                        " ".join(sample["tokens"]),
                        " ".join(sample["labels"]),
                    ]
                    for sample in data
                ]
                self.df = pd.DataFrame(dataset, columns=["id", "sentence", "labels"])
            else:
                dataset = [
                    [sample["id"], " ".join(sample["tokens"])] for sample in data
                ]
                self.df = pd.DataFrame(dataset, columns=["id", "sentence"])

    def write_to_csv(self):
        self.df.to_csv(self.output_file, index=False)


legal_dev_data = Dataset(f"{sys.argv[1]}/legal/scidr_dev.json")
legal_train_data = Dataset(f"{sys.argv[1]}/legal/scidr_train.json")
scientific_dev_data = Dataset(f"{sys.argv[1]}/scientific/scidr_dev.json")
scientific_train_data = Dataset(f"{sys.argv[1]}/scientific/scidr_train.json")
