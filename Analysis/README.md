## Different Model Precictions Visualizer

To see the difference in the predictions b/w 2 models and for predictions analysis and evaluation, we use the `models_visualizer.py` file. It takes 2 input csv files which formed by the merging the input_data.json file with the predictions.json file.

After creating the csv files for the models we wish to compare we can visualize the difference in the predictions of each model as follows:

```bash
$ python3 models_visualizer.py <model_1_csv> <model_2_csv>
```
