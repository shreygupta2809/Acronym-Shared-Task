# Shared Task 2: Acronym Disambiguation

This task aims to distinguish the correct expansion for a given acronym

# Dataset

The data folder contains data for English (Legal domain and Scientific domain). The corresponding folder for each language contains two original data files:

- **train.json**: The training samples for acronym disambiguation task. Each sample has three attributes:
  - sentence: The string value of the sample text
  - acronym: The string value of the acronym in the sentence to be disambiguated.
  - label: The string value of the correct expansion of the acronym to be disambiguated.
  - ID: The unique ID of the sample
- **dev.json**: The development set for acronym disambiguation task. The samples in `dev.json` have the same attributes as the samples in `train.json`.

## Reformatting

The data in json format is converted to csv format for better readability and usability for the model. 

```bash
$ python3 utils/convert_to_csv.py
```

This changes the train and dev files for legal and scientific domains to the required csv file. This file also replaces any whitespacing in the acronyms in the training data and diction with `-` to prevent it from getting tokenized separately and causing an error. These files are then provided to the model as input. Note that the final train.csv file for legal domain contained some errors in the acronym and their corresponding matches in the sentence and diction. These changes have been taken care of manually.

# Code

## Baseline

In order to familiarize the participants with this task, the organizers of the shared task provide a rule-based model in the `utils` directory called `baseline.py`. In this baseline, for each sample, the acronym is taken and similarity is computed. The model iterates through the diction list for the acronym and calculates the number of overlapping words in the sample sentence with the candidate long form. The highest scoring similarity candidate is chosen as the final prediction. To run this model, use the following command:

```bash
$ python3 utils/baseline.py -input <path/to/input.json> -output <path/to/output.json>
```
The outputs have been stored in outputs/

## Utils

- `scrape-wiki-article.ipynb`: This file contains the web scraping process. It uses the provided diction file as keywords and with the help of wikipedia library, gets the articles corresponding to the data. The final set is saved in wiki_article.txt and used for fine-tuning.

- `lm-fine-tuning.ipynb`: This file contains the fine-tuning process for the language model. It takes in the scraped WikiPedia articles and uses it to run the Mask-Language Modeling (`utils/mlm.py`) approach for fine-tuning on the model provided. The configs are stored at every checkpoint which are later used in the final architecture (`models/model configs/finetune_scibert_config.json`).

- `json_to_csv.py`: This file takes a given csv file and converts it to json format as required by the scorer.py file.

## Our Model
The code for running the final architecture is contained within `notebooks/acronym_disambiguation.ipynb` file. 
The files generated after data processing are in `data/domain/combined_acronym_dict.json`. The most appropriate expansion key is selected from this dict and stored in `data/domain/fix_combined_acronym_dict.json`.

### Instructions to run:
1. Open `notebooks/acronym_disambiguation.ipynb` and load the GPUs 
2. Set the appropriate file name in the `config` class
3. Run all cells 
4. Models in each fold will be saved as `model_{fold}` and prediction on dev set will be stored in `output.json`

This file contains sections corresponding to the data loading, processing, model training, validation and final predictions.

# Evaluation
To evaluate the predictions, run the command:

```bash
$ python3 utils/scorer.py -g path/to/gold.json -p path/to/predictions.json -v[optional]
```
The path/to/gold.json and path/to/predictions.json should be replaced with the real paths to the gold file (e.g., data/domain/dev.json for evaluation on development set) and predictions file. The official evaluation metrics are the macro-averaged precision, recall and F1 for correct long-form predictions. For verbose evaluation (including the micro-averaged precision, recall and F1 and also the accuracy of the predictions), include the -v flag in the command.

Here domain are legal and scientific.