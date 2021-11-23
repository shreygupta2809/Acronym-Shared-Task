from ast import literal_eval
import sys
import pandas as pd

model_1_preds_path = sys.argv[1]
model_2_preds_path = sys.argv[2]

model_1_df = pd.read_csv(
    model_1_preds_path,
    usecols=["id", "tokens", "labels", "predictions"],
    converters={
        "tokens": literal_eval,
        "labels": literal_eval,
        "predictions": literal_eval,
    },
)

model_2_df = pd.read_csv(
    model_2_preds_path,
    usecols=["id", "tokens", "labels", "predictions"],
    converters={
        "tokens": literal_eval,
        "labels": literal_eval,
        "predictions": literal_eval,
    },
)

model_1_df["id"] = model_1_df["id"].astype(int)
model_2_df["id"] = model_2_df["id"].astype(int)


wrong_model_1_df = model_1_df[model_1_df["labels"] != model_1_df["predictions"]]
wrong_model_2_df = model_2_df[model_2_df["labels"] != model_2_df["predictions"]]

model1_ids = set(wrong_model_1_df["id"])
model2_ids = set(wrong_model_2_df["id"])

differences = list(model1_ids.symmetric_difference(model2_ids))

df = model_1_df[model_1_df["id"].isin(differences)].reset_index().drop(columns="index")
df = pd.merge(df, model_2_df, how="inner", on="id", suffixes=("_model_1", "_model_2"))
df["model_1_wrong"] = df["labels_model_1"] != df["predictions_model_1"]
df["model_2_wrong"] = df["labels_model_2"] != df["predictions_model_2"]

model_1_name = model_1_preds_path.split("/")
model_1_name = model_1_name[0] + "_" + model_1_name[-1].split(".")[0]

model_2_name = model_2_preds_path.split("/")
model_2_name = model_2_name[0] + "_" + model_2_name[-1].split(".")[0]
df.to_csv(f"{model_1_name}-{model_2_name}.csv")
