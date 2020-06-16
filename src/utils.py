import re
import numpy
import pandas
from sklearn import preprocessing

from src import data_fetching
from src.config import *

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import hunspell
import swifter

from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

from imblearn.over_sampling import RandomOverSampler


def untokenize_sentence(simple_list):
    return TreebankWordDetokenizer().detokenize(simple_list)


def encode_label(list_of_labels):
    encoder = preprocessing.LabelEncoder()
    return encoder.fit_transform(list_of_labels)


def print_f1_score(predictions, labels, confusion_matrix=CONFUSION_MATRIX, prec_and_rec=PREC_AND_RECALL):
    preds = pd.Series(map(int, predictions), name='Predicted')
    labs = pd.Series(map(int, labels), name='Actual')
    if confusion_matrix:
        print(pd.crosstab(labs, preds))
    precision = precision_score(labs, preds, average='macro')
    recall = recall_score(labs, preds, average='macro')
    score = f1_score(labs, preds, average='macro')  # 2*(precision*recall)/(precision+recall)
    print("F1-SCORE: " + str(score))
    if prec_and_rec:
        print("Recall: " + str(recall))
        print("Precision: " + str(precision))
    return score


def get_averaged_predictions(predictions_array, length, label_dict):
    averaged_predictions = pd.DataFrame(0, index=numpy.arange(length), columns=label_dict)
    for predictions in predictions_array:
        averaged_predictions = averaged_predictions.add(predictions)
    averaged_predictions = (averaged_predictions/len(predictions_array)).idxmax(axis=1)
    return averaged_predictions


def decode_label(predictions_array):
    labels = ['N', 'NONE', 'NEU', 'P']
    result = [labels[one_prediction] for one_prediction in predictions_array]
    return result


# AUXILIAR
def csv2ftx(data, labels, sLang, sPhase, folder, filename_ending=''):
    result = pandas.DataFrame()
    if filename_ending != '':
        filename_ending = '_' + filename_ending
    result['labels'] = ['__label__' + str(label) for label in labels]
    result['content'] = data
    result.to_csv('../dataset/{}/intertass_{}_{}{}.txt'.format(folder, sLang, sPhase, filename_ending), header=None, index=None, sep=' ')
    return


def print_separator(string_for_printing):
    print()
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print("////                                    " + string_for_printing)
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()


if __name__ == '__main__':
    for sLang in ['es', 'cr', 'mx', 'pe', 'uy']:
        print('------->>>>>> {}'.format(sLang))
        print(data_fetching.get_dataframe_from_ftx_format(sLang, 'back', train_plus_dev=True))
