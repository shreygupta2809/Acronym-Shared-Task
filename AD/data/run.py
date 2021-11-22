import pandas as pd
import json
import os
import sys
# import pickle

class Dataset():
    def __init__(self, filename):
        self.filename = filename
        self.output_file = self.filename.split('.')[0]+'.csv'
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


    def write_to_csv(self):
        self.df.to_csv(self.output_file,index=False)

    def show_csv(self):
        print(pd.read_csv(self.output_file).head())

# dev_data = Dataset(f'dev.json')
train_data = Dataset(f'test.json')
# with open('data.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data)