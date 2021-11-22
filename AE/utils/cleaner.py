import json
import os
import sys
import re

regex = re.compile(r"[^0-9a-zA-Z]")


def remove_characters(data_json):
    new_data_json = []
    for record in data_json:
        new_record = {"id": record["id"], "tokens": [], "labels": []}
        for index, tok in enumerate(record["tokens"]):
            if regex.match(tok):
                if record["labels"][index] in ["O", "I-long"]:
                    continue
                elif record["labels"][index] == "B-long":
                    if (
                        index < len(record["labels"]) - 1
                        and record["labels"][index + 1] == "I-long"
                    ):
                        record["labels"][index + 1] = "B-long"
                    continue
                else:
                    pass
            new_record["tokens"].append(record["tokens"][index])
            new_record["labels"].append(record["labels"][index])
        new_data_json.append(new_record)
    return new_data_json


with open("data/english/legal/scidr_train.json", "r") as f:
    train_legal_data = json.load(f)

with open("data/english/legal/scidr_dev.json", "r") as f:
    dev_legal_data = json.load(f)

with open("data/english/scientific/scidr_train.json", "r") as f:
    train_scientific_data = json.load(f)

with open("data/english/scientific/scidr_dev.json", "r") as f:
    dev_scientific_data = json.load(f)

train_legal_data = remove_characters(train_legal_data)
dev_legal_data = remove_characters(dev_legal_data)
train_scientific_data = remove_characters(train_scientific_data)
dev_scientific_data = remove_characters(dev_scientific_data)

os.mkdir(sys.argv[1])
os.mkdir(f"{sys.argv[1]}/legal")
os.mkdir(f"{sys.argv[1]}/scientific")

with open(f"{sys.argv[1]}/legal/scidr_dev.json", "w") as f:
    json.dump(dev_legal_data, f, indent=2)

with open(f"{sys.argv[1]}/legal/scidr_train.json", "w") as f:
    json.dump(train_legal_data, f, indent=2)

with open(f"{sys.argv[1]}/scientific/scidr_dev.json", "w") as f:
    json.dump(dev_scientific_data, f, indent=2)

with open(f"{sys.argv[1]}/scientific/scidr_train.json", "w") as f:
    json.dump(train_scientific_data, f, indent=2)

os.system(f"python3 utils/dataset_generator.py {sys.argv[1]}")
os.system(f"python3 utils/dataset_reformatter.py {sys.argv[1]}")
# os.system(f"zip -r {sys.argv[1]}.zip {sys.argv[1]}")
