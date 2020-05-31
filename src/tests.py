from gensim.models import KeyedVectors
import pandas as pd
import utils
from sklearn import svm, preprocessing, linear_model
import os
import re
import string
import hunspell
import fasttext
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from tweet_preprocessing import preprocess

from pathlib import Path

# from wikipedia2vec import Wikipedia2Vec

# Wikipedia2Vec.load('./embeddings/eswiki_20180420_300d.pkl')

sLang = 'es'
train_data, dev_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for lang in ['es', 'cr', 'mx', 'pe', 'uy']:
    train, dev, test, _ = utils.read_files(lang)
    if lang == sLang:
        dev_data = dev
        test_data = test
    else:
        train_data = pd.concat([train_data, train], ignore_index=True).reset_index(drop=True)

print(train_data)
print(dev_data)
print(test_data)




