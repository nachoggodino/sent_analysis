from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import hunspell
import pandas as pd
from flair.visual.training_curves import Plotter


from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings, BertEmbeddings, \
    ELMoEmbeddings, FastTextEmbeddings, BytePairEmbeddings, XLNetEmbeddings
from flair.training_utils import EvaluationMetric

from src import config, utils, data_fetching, tweet_preprocessing
from src.config import *

import spacy
from flair.embeddings import WordEmbeddings
from flair.data import Sentence


def predict_with_bert_model(model, data, label_dictionary):
    probabilities_list = list()
    for text in data:
        row_prob = dict()
        result = model.predict(Sentence(text), multi_class_prob=True)
        for label in zip(result[0].labels):
            row_prob.update({label[0].value: label[0].score})
        probabilities_list.append(row_prob)
    final_df = pd.DataFrame(probabilities_list)
    return final_df, final_df[label_dictionary].idxmax(axis=1)


if __name__ == '__main__':

    bStoreFiles = True

    bLibreOffice = False
    bLemmatize = False
    bTokenize = False
    bPreprocess = True
    bBackTranslation = False
    bTrainPlusDev = True
    bUpsampling = True
    bCrossLingual = True

    sPreprocessFiles = 'allPrep'

    model_number = 'june_test1'

    if bLibreOffice:
        print("Loading Hunspell directory")
        dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")

    if bLemmatize:
        print("Loading Spacy Model")
        lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

    for S_DATASET in DATASET_ARRAY:

        print('################    DATASET: {}    ###############################'.format(S_DATASET))
        print()

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if bUpsampling:
            print('Upsampling the data...')
            train_data = utils.perform_upsampling(train_data)

        if bPreprocess:
            train_data['content'] = tweet_preprocessing.preprocess(train_data['content'], bLowercasing=True,
                                                                   bPunctuation=True, bAll=False)
            dev_data['content'] = tweet_preprocessing.preprocess(dev_data['content'], bLowercasing=True,
                                                                 bPunctuation=True, bAll=False)
            test_data['content'] = tweet_preprocessing.preprocess(test_data['content'], bLowercasing=True,
                                                                  bPunctuation=True, bAll=False)

        if bTokenize:
            print("Tokenizing...")
            train_data['content'] = train_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.progress_bar(False).apply(
                    lambda row: utils.tokenize_sentence(row.content), axis=1)

        if bLibreOffice:
            print("LibreOffice Processing... ")
            train_data['content'] = train_data.swifter.progress_bar(True).apply(
                lambda row: utils.libreoffice_processing(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.apply(
                lambda row: utils.libreoffice_processing(row.content), axis=1)
            test_data['content'] = test_data.swifter.apply(
                    lambda row: utils.libreoffice_processing(row.content), axis=1)

        if bLemmatize:
            print("Lemmatizing data...")
            train_data['content'] = train_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)
            dev_data['content'] = dev_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)
            test_data['content'] = test_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.content, lemmatizer), axis=1)

        if bTokenize:
            train_data['content'] = [utils.untokenize_sentence(sentence) for sentence in train_data['content']]
            dev_data['content'] = [utils.untokenize_sentence(sentence) for sentence in dev_data['content']]
            test_data['content'] = [utils.untokenize_sentence(sentence) for sentence in test_data['content']]

        if bStoreFiles:
            utils.csv2ftx(train_data.content, train_data.sentiment, S_DATASET, 'train', 'flair')
            utils.csv2ftx(dev_data.content, dev_data.sentiment, S_DATASET, 'dev', 'flair')
            utils.csv2ftx(test_data.content, test_data.sentiment, S_DATASET, 'test', 'flair')

        corpus = Corpus = ClassificationCorpus('./dataset/flair/', train_file='intertass_{}_train.txt'.format(S_DATASET),
                                               dev_file='intertass_{}_dev.txt'.format(S_DATASET),
                                               test_file='intertass_{}_test.txt'.format(S_DATASET))

        # word_embeddings = [BertEmbeddings('bert-base-multilingual-cased')]
        word_embeddings = [BertEmbeddings('dccuchile/bert-base-spanish-wwm-cased')]

        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, rnn_type='LSTM',
                                                    reproject_words_dimension=256, dropout=0.35, rnn_layers=2)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train('./beto/dev/models/{}/{}/'.format(BERT_MODEL_NAME, S_DATASET), max_epochs=20, mini_batch_size=32, anneal_factor=0.5,
                      train_with_dev=bTrainPlusDev, learning_rate=0.05, patience=1, )

        best_model = TextClassifier.load('./beto/dev/models/{}/{}/best-model.pt'.format(BERT_MODEL_NAME, S_DATASET))

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      DEV')
        _, dev_predictions = predict_with_bert_model(best_model, dev_data.content, label_dictionary)
        utils.print_confusion_matrix(dev_predictions, utils.encode_label(dev_data['sentiment']))

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      TEST')
        _, test_predictions = predict_with_bert_model(best_model, dev_data.content, label_dictionary)
        utils.print_confusion_matrix(test_predictions, utils.encode_label(test_data['sentiment']))

        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
