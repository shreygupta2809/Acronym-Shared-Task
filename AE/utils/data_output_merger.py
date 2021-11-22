import json
import sys
import pandas as pd

for DOMAIN in ["legal", "scientific"]:
    for _data in ["dev", "train"]:
        with open(f"{sys.argv[1]}/{DOMAIN}/scidr_{_data}.json", "r") as f:
            data = json.load(f)

        with open(f"{sys.argv[1]}/{DOMAIN}/{_data}_out.json", "r") as f:
            predictions = json.load(f)

        df_data = pd.DataFrame(data)
        df_preds = pd.DataFrame(predictions)
        df = pd.merge(left=df_data, right=df_preds, how="inner", on="id")
        df = df.rename(columns={"labels_x": "labels"})
        df.drop(columns="labels_y", inplace=True)

        df.to_csv(f"{sys.argv[1]}/{DOMAIN}_{_data}.csv")


# for DOMAIN in ["legal", "scientific"]:
#     for _data in ["dev", "train"]:
#         with open(f"data/english/{DOMAIN}/scidr_{_data}.json", "r") as f:
#             data = json.load(f)

#         with open(f"output/english/{DOMAIN}/out_scidr_{_data}.json", "r") as f:
#             predictions = json.load(f)

#         df_data = pd.DataFrame(data)
#         df_preds = pd.DataFrame(predictions)
#         df = pd.merge(left=df_data, right=df_preds, how="inner", on="id")
#         df = df.rename(columns={"labels_x": "labels"})
#         df.drop(columns="labels_y", inplace=True)

#         df.to_csv(f"{DOMAIN}_{_data}.csv")
