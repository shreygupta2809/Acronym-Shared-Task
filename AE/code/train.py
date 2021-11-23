import os
import sys

import json
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

sys.path.append("../")
from bert_sklearn import BertTokenClassifier, load_model

#################################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--domain", type=str, default='leg', dest='domain', help="datatset domain ('legal' or 'scientific')")
parser.add_argument("--model-tag", type=str, default='untitled', dest='model_tag', help="base model tag for saving model and related files")
parser.add_argument("--data-dir", type=str, default='', dest='datadir', help="path to data directory")
parser.add_argument("--data-subdir", type=str, default='scibert_sduai', dest='datasubdir', help="path to data subdirectory")
parser.add_argument("--epochs", type=int, default=3, dest='epochs', help="number of training epochs")
parser.add_argument("--seed", type=int, default=0, dest='seed', help="seed")
parser.add_argument("--mlp-hidden-dim", type=int, default=500, dest='mlp_hidden_dim', help="mlp hidden layer dimensionality")
parser.add_argument("--mlp-hidden-layers", type=int, default=0, dest='mlp_hidden_layers', help="mlp hidden layer count")
parser.add_argument("--lr", type=float, default=1e-4, dest='lr', help="learning rate")
parser.add_argument("--validation-fraction", type=float, default=0.1, dest='validation_fraction', help="validation fraction")
parser.add_argument("--train-batch-size", type=int, default=16, dest='train_batch_size', help="train batch size")
parser.add_argument("--eval-batch-size", type=int, default=16, dest='eval_batch_size', help="eval batch size")
parser.add_argument("--train-gold-file", type=str, default='scidr_train.json', dest='train_gold_file', help="path to train gold file (json)")
parser.add_argument("--dev-gold-file", type=str, default='scidr_dev.json', dest='dev_gold_file', help="path to dev gold file (json)")
parser.add_argument("--base-model", type=str, default='bert-base-cased', dest='base_model', help="base encoder model")

args = parser.parse_args()

#################################################################################################################################################

#======================================================================================

def flatten(l):
    return [item for sublist in l for item in sublist]

def bioless(l):
    return l.replace('B-','').replace('I-','')

def read_CoNLL2003_format(filename, idx=3):
    """Read file in CoNLL-2003 shared task format""" 
    
    lines =  open(filename).read().strip()
    
    # find sentence-like boundaries
    lines = lines.split("\n\n")  
    
     # split on newlines
    lines = [line.split("\n") for line in lines]
    
    # get tokens
    tokens = [[l.split()[0] for l in line] for line in lines]
    
    # get labels/tags
    labels = [[bioless(l.split()[idx]) for l in line] for line in lines]
    
    data= {'tokens': tokens, 'labels': labels}
    df=pd.DataFrame(data=data)
    
    return df


def get_conll2003_data(trainfile=os.path.join(args.datadir,args.datasubdir,"train.txt"),
                  devfile=os.path.join(args.datadir,args.datasubdir,"dev.txt")):

    train = read_CoNLL2003_format(trainfile)
    print("Train data: %d sentences, %d tokens"%(len(train),len(flatten(train.tokens))))

    dev = read_CoNLL2003_format(devfile)
    print("Dev data: %d sentences, %d tokens"%(len(dev),len(flatten(dev.tokens))))

    return train, dev

train, dev = get_conll2003_data()

X_train, y_train = train.tokens, train.labels
X_dev, y_dev = dev.tokens, dev.labels

label_list = np.unique(flatten(y_train))
label_list = list(label_list)
print("\nNER tags:",label_list)

print(train.head())

#======================================================================================

# define model
#model = BertTokenClassifier(bert_model='scibert-scivocab-cased',
#model = BertTokenClassifier(bert_model='bert-base-cased',
#model = BertTokenClassifier(bert_model='bert-large-cased',
model = BertTokenClassifier(bert_model=args.base_model,
                            random_state=args.seed,
                            epochs=args.epochs,
                            learning_rate=args.lr,
                            train_batch_size=args.train_batch_size,
                            eval_batch_size=args.eval_batch_size,
                            validation_fraction=args.validation_fraction,  
                            num_mlp_hiddens=args.mlp_hidden_dim,
                            num_mlp_layers=args.mlp_hidden_layers,                          
                            label_list=label_list,
                            ignore_label=['O'])

print("Bert wordpiece tokenizer max token length in train: %d tokens"% model.get_max_token_len(X_train))
print("Bert wordpiece tokenizer max token length in dev: %d tokens"% model.get_max_token_len(X_dev))

model.max_seq_length = 272
model.gradient_accumulation_steps = 2
print(model)

#======================================================================================

# finetune model on train data
model.fit(X_train, y_train)

#======================================================================================

# print report on classifier stats
print(classification_report(flatten(y_dev), flatten(model.predict(X_dev))))

f1_dev = model.score(X_dev, y_dev, average='macro')
print("Dev f1: %0.02f"%(f1_dev))

#======================================================================================

#save model
savepath = f"scidr_eng_{args.domain}_{args.model_tag}.bin"
model.save(savepath)

#======================================================================================

def fixed_tags(tags):
    fixed = []
    cont = None
    for tag in tags:
        if tag == 'O':
            fixed.append(tag)
            cont = None
        else:
            if cont == tag:
                fixed.append("I-"+tag)
            else: 
                fixed.append("B-"+tag)
                cont = tag
    assert len(list(filter(lambda x: 'long' in x,fixed)))== len(list(filter(lambda x:'long' in x,tags)))
    assert len(list(filter(lambda x: 'short' in x,fixed)))== len(list(filter(lambda x:'short' in x,tags)))

    assert len(fixed) == len(tags)
    return fixed


y_preds = model.predict(X_train)

#======================================================================================

submit = pd.read_json(os.path.join(args.datadir,args.train_gold_file))
submit.head()

len(submit)

submit['predictions'] = [fixed_tags(y_pred) for y_pred in y_preds]
# print(submit.groupby('predictions').count())
submit.drop('tokens', axis=1, inplace=True)
submit["id"] = submit['id'].astype(str)
submit.head()

train_pred_file = f'scidr_{args.model_tag}_{args.domain}_{args.base_model}_train_out.json'
submit.to_json(train_pred_file, orient = 'records') # Download output file for training set

#======================================================================================

y_preds = model.predict(X_dev)
submit = pd.read_json(os.path.join(args.datadir,args.dev_gold_file))
submit['predictions'] = [fixed_tags(y_pred) for y_pred in y_preds]
# print(submit.groupby('predictions').count())
submit.drop('tokens', axis=1, inplace=True)
submit["id"] = submit['id'].astype(str)
submit.head()

dev_pred_file = f'scidr_{args.model_tag}_{args.domain}_{args.base_model}_dev_out.json'
submit.to_json(dev_pred_file, orient = 'records') # Download output file for dev set

#======================================================================================

def run_evaluation(gold_path,pred_path,verbose=True):
    with open(gold_path) as file:
        gold = dict([(d["id"], d["labels"]) for d in json.load(file)])
    with open(pred_path) as file:
        pred = dict([(d["id"], d["predictions"]) for d in json.load(file)])
        pred = [pred[k] for k, v in gold.items()]
        gold = [gold[k] for k, v in gold.items()]
    p, r, f1 = score_phrase_level(gold, pred, verbos=verbose)
    return p, r, f1


def score_phrase_level(key, predictions, verbos=False):
    gold_shorts = set()
    gold_longs = set()
    pred_shorts = set()
    pred_longs = set()

    def find_phrase(seq, shorts, longs):

        for i, sent in enumerate(seq):
            short_phrase = []
            long_phrase = []
            for j, w in enumerate(sent):
                if "B" in w or "O" in w:
                    if len(long_phrase) > 0:
                        longs.add(
                            str(i)
                            + "-"
                            + str(long_phrase[0])
                            + "-"
                            + str(long_phrase[-1])
                        )
                        long_phrase = []
                    if len(short_phrase) > 0:
                        shorts.add(
                            str(i)
                            + "-"
                            + str(short_phrase[0])
                            + "-"
                            + str(short_phrase[-1])
                        )
                        short_phrase = []
                if "short" in w:
                    short_phrase.append(j)
                if "long" in w:
                    long_phrase.append(j)
            if len(long_phrase) > 0:
                longs.add(
                    str(i) + "-" + str(long_phrase[0]) + "-" + str(long_phrase[-1])
                )
            if len(short_phrase) > 0:
                shorts.add(
                    str(i) + "-" + str(short_phrase[0]) + "-" + str(short_phrase[-1])
                )

    find_phrase(key, gold_shorts, gold_longs)
    find_phrase(predictions, pred_shorts, pred_longs)

    def find_prec_recall_f1(pred, gold):
        correct = 0
        for phrase in pred:
            if phrase in gold:
                correct += 1
        prec = correct / len(pred) if len(pred) > 0 else 1
        recall = correct / len(gold) if len(gold) > 0 else 1
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
        return prec, recall, f1

    prec_short, recall_short, f1_short = find_prec_recall_f1(pred_shorts, gold_shorts)
    prec_long, recall_long, f1_long = find_prec_recall_f1(pred_longs, gold_longs)
    precision_micro, recall_micro, f1_micro = find_prec_recall_f1(
        pred_shorts.union(pred_longs), gold_shorts.union(gold_longs)
    )

    precision_macro = (prec_short + prec_long) / 2
    recall_macro = (recall_short + recall_long) / 2
    f1_macro = (
        2 * precision_macro * recall_macro / (precision_macro + recall_macro)
        if precision_macro + recall_macro > 0
        else 0
    )

    if verbos:
        print(
            "Shorts: P: {:.2%}, R: {:.2%}, F1: {:.2%}".format(
                prec_short, recall_short, f1_short
            )
        )
        print(
            "Longs: P: {:.2%}, R: {:.2%}, F1: {:.2%}".format(
                prec_long, recall_long, f1_long
            )
        )
        print(
            "micro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}".format(
                precision_micro, recall_micro, f1_micro
            )
        )
        print(
            "macro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}".format(
                precision_macro, recall_macro, f1_macro
            )
        )

    return precision_macro, recall_macro, f1_macro

print('\nTRAIN EVAL')
print('='*70)
train_p, train_r, train_f1 = run_evaluation(os.path.join(args.datadir,args.train_gold_file),train_pred_file)

print('\nDEV EVAL')
print('='*70)
dev_p, dev_r, dev_f1 = run_evaluation(os.path.join(args.datadir,args.dev_gold_file),dev_pred_file)




