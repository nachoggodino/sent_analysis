import fasttext
from src import utils, config, data_fetching, tweet_preprocessing
from src.config import *
import pandas as pd


def predict_with_fasttext_model(model, data, label_dictionary):
    probabilities_list = list()
    for text in data:
        row_prob = dict()
        result = model.predict(text, k=len(list(label_dictionary)))
        for label, prob in zip(result[0], result[1]):
            row_prob.update({label[-1]: prob})
        probabilities_list.append(row_prob)
    final_df = pd.DataFrame(probabilities_list)
    ft_predictions = [int(model.predict(text)[0][0][-1]) for text in data]
    return final_df, ft_predictions


if __name__ == '__main__':

    for S_DATASET in DATASET_ARRAY:

        if B_COMMONCRAWL:
            PRETRAINED_VECTORS_PATH = '../resources/embeddings/cc.es.300.vec'
            DIMENSION = 300
        elif B_WIKIPEDIA:
            PRETRAINED_VECTORS_PATH = '../resources/embeddings/wiki.es.vec'
            DIMENSION = 300
        elif B_WIKIPEDIA_ALIGNED:
            PRETRAINED_VECTORS_PATH = '../resources/embeddings/wiki.es.align.vec'
            DIMENSION = 300
        elif B_INGEOTEC:
            PRETRAINED_VECTORS_PATH = '../resources/embeddings/ingeotec_embeddings/es-{}-100d/es-{}-100d.vec' \
                .format('ES'.upper(), 'ES'.upper())
            DIMENSION = 100

        print('################    DATASET: {}    ###############################'.format(S_DATASET))
        print()

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if B_UPSAMPLING:
            print('Performing upsampling...')
            train_data = tweet_preprocessing.perform_upsampling(train_data)

        # PRE-PROCESSING
        print('Data preprocessing...')
        train_data['preprocessed'] = tweet_preprocessing.preprocess_data(train_data['content'], 'embedding')
        dev_data['preprocessed'] = tweet_preprocessing.preprocess_data(dev_data['content'], 'embedding')
        if B_TEST_PHASE is True:
            test_data['preprocessed'] = tweet_preprocessing.preprocess_data(test_data['content'], 'embedding')

        utils.csv2ftx(train_data.preprocessed, train_data.sentiment, S_DATASET, 'train', 'ftx')
        utils.csv2ftx(dev_data.preprocessed, dev_data.sentiment, S_DATASET, 'dev', 'ftx')
        utils.csv2ftx(test_data.preprocessed, test_data.sentiment, S_DATASET, 'test', 'ftx')

        model = fasttext.train_supervised(input='../dataset/{}/intertass_{}_train.txt'.format('ftx', S_DATASET),
                                          pretrained_vectors=PRETRAINED_VECTORS_PATH,
                                          lr=FT_LEARNING_RATE, epoch=FT_EPOCH, wordNgrams=FT_WORDGRAM, seed=1234,
                                          dim=DIMENSION, verbose=5)

        model.save_model(path='../fasttext/models/{}_{}'.format(S_DATASET, FT_MODEL_NAME))

        print(len(model.words))

        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in dev_data['content']]
        print('DEV')
        utils.print_f1_score(predictions, dev_data.sentiment)
        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in test_data['content']]
        print('TEST')
        utils.print_f1_score(predictions, test_data.sentiment)

        print('-----------------------------------NEXT---------------------------------------------------')
