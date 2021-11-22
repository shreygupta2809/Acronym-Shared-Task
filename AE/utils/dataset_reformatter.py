import os
import sys

import nltk
import pandas as pd

directory = sys.argv[1]


def reformat_test(x, test=False):
    tok_text = nltk.word_tokenize(x["sentence"])
    word_pos = nltk.pos_tag(tok_text)
    return "\n".join([f"{word} {pos} O O" for (word, pos) in word_pos])


def reformat(x):
    tok_text = nltk.word_tokenize(x["sentence"])
    tags = x["labels"].split()
    word_pos = nltk.pos_tag(tok_text)
    return "\n".join([f"{word} {pos} O {i}" for (word, pos), i in zip(word_pos, tags)])


def make_data(filename):
    df = pd.read_csv(f"{directory}/{filename}.csv")
    filename = filename.split("/")
    filename[1] = filename[1].split("_")[-1]

    if filename == "test":
        df["reformatted_data"] = df[["sentence"]].apply(reformat_test, axis=1)
    else:
        df["reformatted_data"] = df[["sentence", "labels"]].apply(reformat, axis=1)

    with open(
        f"{directory}/{filename[0]}/scibert_sduai/{filename[1]}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n\n".join(df["reformatted_data"].tolist()))


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    os.system(f"mkdir -p {sys.argv[1]}/legal/scibert_sduai")
    os.system(f"mkdir -p {sys.argv[1]}/scientific/scibert_sduai")
    make_data("legal/scidr_train")
    make_data("legal/scidr_dev")
    make_data("scientific/scidr_train")
    make_data("scientific/scidr_dev")
