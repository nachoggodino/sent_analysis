import re

import numpy
import pandas
from sklearn import preprocessing

from src import data_fetching

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import hunspell
import swifter

from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk

from imblearn.over_sampling import RandomOverSampler


def untokenize_sentence(simple_list):
    return TreebankWordDetokenizer().detokenize(simple_list)


def lemmatize_sentence(sentence, lemmatizer):
    data = untokenize_sentence(sentence)
    return [token.lemma_ for token in lemmatizer(data)]


def encode_label(list_of_labels):
    encoder = preprocessing.LabelEncoder()
    return encoder.fit_transform(list_of_labels)


def print_confusion_matrix(predictions, labels, print_confusion_matrix=False, print_prec_and_rec=False):
    preds = pd.Series(map(int, predictions), name='Predicted')
    labs = pd.Series(map(int, labels), name='Actual')
    df_confusion = pd.crosstab(labs, preds)
    if print_confusion_matrix:
        print(df_confusion)
    prec = precision_score(labs, preds, average='macro')
    rec = recall_score(labs, preds, average='macro')
    score = 2*(prec*rec)/(prec+rec)
    print("F1-SCORE: " + str(score))
    if print_prec_and_rec:
        print("Recall: " + str(rec))
        print("Precision: " + str(prec))
    print()
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
    result.to_csv('dataset/{}/intertass_{}_{}{}.txt'.format(folder, sLang, sPhase, filename_ending), header=None, index=None, sep=' ')
    return


if __name__ == '__main__':
    for sLang in ['es', 'cr', 'mx', 'pe', 'uy']:
        print('------->>>>>> {}'.format(sLang))
        print(data_fetching.get_dataframe_from_ftx_format(sLang, 'back', train_plus_dev=True))
