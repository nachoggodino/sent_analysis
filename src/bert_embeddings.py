from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.data import Sentence
from flair.trainers import ModelTrainer
from flair.embeddings import BertEmbeddings
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd

from src import utils, data_fetching, tweet_preprocessing
from src.config import *


def predict_with_bert_model(model, data, label_dictionary):
    probabilities_list = list()
    for text in tqdm(data, desc="Predicting with BERT"):
        row_prob = dict()
        result = model.predict(Sentence(text), multi_class_prob=True)
        for label in zip(result[0].labels):
            row_prob.update({label[0].value: label[0].score})
        probabilities_list.append(row_prob)
    final_df = pd.DataFrame(probabilities_list)
    return final_df, final_df[label_dictionary].idxmax(axis=1)


if __name__ == '__main__':

    for S_DATASET in DATASET_ARRAY:

        print('################    DATASET: {}    ###############################'.format(S_DATASET))
        print()

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if B_UPSAMPLING:
            print('Upsampling the data...')
            train_data = tweet_preprocessing.perform_upsampling(train_data)

        # PRE-PROCESSING
        print('Data preprocessing...')
        train_data['preprocessed'] = tweet_preprocessing.preprocess_data(train_data['content'], 'embedding')
        dev_data['preprocessed'] = tweet_preprocessing.preprocess_data(dev_data['content'], 'embedding')
        if B_TEST_PHASE is True:
            test_data['preprocessed'] = tweet_preprocessing.preprocess_data(test_data['content'], 'embedding')

        utils.csv2ftx(train_data.content, train_data.sentiment, S_DATASET, 'train', 'flair')
        utils.csv2ftx(dev_data.content, dev_data.sentiment, S_DATASET, 'dev', 'flair')
        utils.csv2ftx(test_data.content, test_data.sentiment, S_DATASET, 'test', 'flair')

        corpus = Corpus = ClassificationCorpus('../dataset/flair/',
                                               train_file='intertass_{}_train.txt'.format(S_DATASET),
                                               dev_file='intertass_{}_dev.txt'.format(S_DATASET),
                                               test_file='intertass_{}_test.txt'.format(S_DATASET))

        # class_weights = compute_class_weight('balanced', [0, 1, 2, 3], y=train_data.sentiment)
        # dict_weights = dict()
        # for i, label in enumerate(class_weights):
        #     dict_weights.update({str(label): class_weights[i]})

        # word_embeddings = [BertEmbeddings('bert-base-multilingual-cased')]
        word_embeddings = [BertEmbeddings('dccuchile/bert-base-spanish-wwm-cased')]

        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                    rnn_type='LSTM', reproject_words_dimension=256, dropout=0.35,
                                                    rnn_layers=2)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)

        trainer = ModelTrainer(classifier, corpus)
        trainer.train('../bert/beto/models/{}/{}/'.format(BERT_MODEL_NAME, S_DATASET), max_epochs=20,
                      mini_batch_size=32, anneal_factor=0.5, train_with_dev=B_TRAIN_PLUS_DEV, learning_rate=0.05,
                      patience=1)

        best_model = TextClassifier.load('../bert/beto/models/{}/{}/best-model.pt'.format(BERT_MODEL_NAME, S_DATASET))

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      DEV')
        _, dev_predictions = predict_with_bert_model(best_model, dev_data.content, label_dictionary)
        utils.print_f1_score(dev_predictions, utils.encode_label(dev_data['sentiment']))

        print('------------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      TEST')
        _, test_predictions = predict_with_bert_model(best_model, test_data.content, label_dictionary)
        utils.print_f1_score(test_predictions, utils.encode_label(test_data['sentiment']))

        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
