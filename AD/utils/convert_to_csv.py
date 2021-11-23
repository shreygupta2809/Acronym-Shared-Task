import pandas as pd
import json
import os
import sys

class Dataset():
    def __init__(self, filename):
        self.filename = filename
        self.output_file = self.filename.split('.json')[0]+'.csv'
        self.convert_to_df()
        self.write_to_csv()
        self.show_csv()

    def convert_to_df(self):
        with open(self.filename) as file:
            data = json.load(file)
            if 'test' in self.filename:
                self.df = pd.DataFrame(data, columns = ['acronym', 'ID', 'sentence'])
                self.df.rename(columns={'acronym': 'acronym_', 'ID':'id', 'sentence': 'text'}, inplace=True)
            else:
                self.df = pd.DataFrame(data, columns = ['acronym', 'label', 'ID', 'sentence'])
                self.df.rename(columns={'acronym': 'acronym_', 'label': 'expansion', 'ID':'id', 'sentence': 'text'}, inplace=True)

    def check(self, x):
        acronym = x[0]
        acronym = acronym.replace(" ","-")
        return acronym

    def write_to_csv(self):
        self.df.acronym_ = self.df.apply(lambda x: self.check(x), axis=1)
        self.df.to_csv(self.output_file,index=False)

    def show_csv(self):
        print(pd.read_csv(self.output_file).head())


if __name__ == '__main__':
    train_data = Dataset(f'../data/legal/train.json')
    dev_data = Dataset(f'../data/legal/dev.json')
    with open("../data/legal/diction.json") as f:
        data = json.load(f)
    for key,val in data.items():
        key = key.replace(" ","-")

    scitrain_data = Dataset(f'../data/scientific/train.json')
    scidev_data = Dataset(f'../data/scientific/dev.json')
    with open("../data/scientific/diction.json") as f:
        data = json.load(f)
    for key,val in data.items():
        key = key.replace(" ","-")