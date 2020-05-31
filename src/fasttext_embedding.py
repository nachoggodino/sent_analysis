import fasttext
from src import utils, config, data_fetching, tweet_preprocessing
from src.config import *

import swifter
import hunspell
import spacy
import pandas as pd




def predict_with_fasttext_model(model, data, label_dictionary):
    probabilities_list = list()
    for text in data:
        row_prob = dict()
        result = model.predict(text, k=len(label_dictionary))
        for label, prob in zip(result[0], result[1]):
            row_prob.update({label[-1]: prob})
        probabilities_list.append(row_prob)
    final_df = pd.DataFrame(probabilities_list)
    ft_predictions = [int(model.predict(text)[0][0][-1]) for text in data]
    return final_df, ft_predictions


if __name__ == '__main__':

    for S_DATASET in DATASET_ARRAY:

        if bCommonCrawl:
            PRETRAINED_VECTORS_PATH = './embeddings/cc.es.300.vec'
            DIMENSION = 300
        elif bWikipedia:
            PRETRAINED_VECTORS_PATH = './embeddings/wiki.es.vec'
            DIMENSION = 300
        elif bWikipediaAligned:
            PRETRAINED_VECTORS_PATH = './embeddings/wiki.es.align.vec'
            DIMENSION = 300
        elif bINGEOTEC:
            PRETRAINED_VECTORS_PATH = './embeddings/ingeotec_embeddings/es-{}-100d/es-{}-100d.vec' \
                .format('ES'.upper(), 'ES'.upper())
            DIMENSION = 100

        print('################    DATASET: {}    ###############################'.format(S_DATASET))
        print()

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if bUpsampling:
            print('Upsampling the data...')
            train_data = tweet_preprocessing.perform_upsampling(train_data)

        if bPreprocess:
            train_data['content'] = tweet_preprocessing.preprocess(
                train_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
                bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
                bUsername=PREP_USERNAME, bURL=PREP_URL)
            dev_data['content'] = tweet_preprocessing.preprocess(
                dev_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
                bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
                bUsername=PREP_USERNAME, bURL=PREP_URL)
            test_data['content'] = tweet_preprocessing.preprocess(
                test_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
                bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
                bUsername=PREP_USERNAME, bURL=PREP_URL)

        if bTokenize:
            print("Tokenizing...")
            train_data['content'] = train_data.swifter.progress_bar(False).apply(
                lambda row: tweet_preprocessing.tokenize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.progress_bar(False).apply(
                lambda row: tweet_preprocessing.tokenize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.progress_bar(False).apply(
                    lambda row: tweet_preprocessing.tokenize_sentence(row.content), axis=1)

        if bLibreOffice:
            print("LibreOffice Processing... ")
            train_data['content'] = train_data.swifter.progress_bar(True).apply(
                lambda row: tweet_preprocessing.libreoffice_processing(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.apply(
                lambda row: tweet_preprocessing.libreoffice_processing(row.content), axis=1)
            test_data['content'] = test_data.swifter.apply(
                    lambda row: tweet_preprocessing.libreoffice_processing(row.content), axis=1)

        if bLemmatize:
            print("Lemmatizing data...")
            train_data['content'] = train_data.swifter.apply(lambda row: tweet_preprocessing.lemmatize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.apply(lambda row: tweet_preprocessing.lemmatize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.apply(lambda row: tweet_preprocessing.lemmatize_sentence(row.content), axis=1)

        if bTokenize:
            train_data['content'] = [utils.untokenize_sentence(sentence) for sentence in train_data['content']]
            dev_data['content'] = [utils.untokenize_sentence(sentence) for sentence in dev_data['content']]
            test_data['content'] = [utils.untokenize_sentence(sentence) for sentence in test_data['content']]

        if bStoreFiles:
            utils.csv2ftx(train_data.content, train_data.sentiment, S_DATASET, 'train', 'ftx')
            utils.csv2ftx(dev_data.content, dev_data.sentiment, S_DATASET, 'dev', 'ftx')
            utils.csv2ftx(test_data.content, test_data.sentiment, S_DATASET, 'test', 'ftx')

        if bTrainModel:
            model = fasttext.train_supervised(input='./dataset/{}/intertass_{}_train.txt'.format(data_folder, S_DATASET),
                                              pretrained_vectors=PRETRAINED_VECTORS_PATH,
                                              lr=FT_LEARNING_RATE, epoch=FT_EPOCH, wordNgrams=FT_WORDGRAM, seed=1234,
                                              dim=DIMENSION, verbose=5)

        if bSaveModel:
            model.save_model(path='./fasttext/models/{}_{}'.format(S_DATASET, MODEL_NAME))

        elif bLoadModel:
            model = fasttext.load_model(path='./fasttext/models/{}_{}'.format(S_DATASET, MODEL_NAME))

        print(len(model.words))

        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in dev_data['content']]
        print('DEV')
        utils.print_confusion_matrix(predictions, dev_data.sentiment)
        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in test_data['content']]
        print('TEST')
        utils.print_confusion_matrix(predictions, test_data.sentiment)

        print('-----------------------------------NEXT---------------------------------------------------')
