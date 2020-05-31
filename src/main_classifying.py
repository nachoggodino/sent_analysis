import fasttext
from flair.models import TextClassifier
from src import config, utils, tweet_preprocessing, data_fetching, fasttext_embedding, bert_embeddings
from src.config import *

from feature_extraction import extract_laughter_feature, extract_exclamation_mark_feature, \
    extract_hashtag_number_feature, extract_length_feature, extract_letter_repetition_feature, \
    extract_question_mark_feature, extract_sent_words_feature, extract_uppercase_feature
from tweet_preprocessing import lemmatize_sentence, libreoffice_processing, perform_upsampling, remove_accents, \
    remove_stopwords, tokenize_sentence, preprocess

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import linear_model, naive_bayes, metrics, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import numpy

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def perform_count_vectors(train_set, dev_set, print_oovs=False, classic_bow=False):
    train = [utils.untokenize_sentence(sentence) for sentence in train_set]
    dev = [utils.untokenize_sentence(sentence) for sentence in dev_set]
    print("Performing CountVectors")
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=classic_bow, min_df=1, lowercase=False)
    count_vect.fit(train)
    if print_oovs:
        dev_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', binary=classic_bow)
        dev_vect.fit(dev)
        oovs = [word for word in dev_vect.vocabulary_ if word not in count_vect.vocabulary_]
        print(oovs)
        print("Length of the vocabulary: {}".format(len(count_vect.vocabulary_)))
        print("OOVS: {} ({}% of the tested vocabulary)".format(len(oovs), len(oovs)*100/len(dev_vect.vocabulary_)))
        print()

    return count_vect.transform(train), count_vect.transform(dev)


def perform_tf_idf_vectors(train_set, dev_set):
    train = [utils.untokenize_sentence(sentence) for sentence in train_set]
    dev = [utils.untokenize_sentence(sentence) for sentence in dev_set]
    print("word level tf-idf")
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train)

    return tfidf_vect.transform(train), tfidf_vect.transform(dev)


def train_model(classifier, x_train, y_train, x_test, y_test, reduced=False, description=''):
    if reduced:
        threshold = 1 if y_train.value_counts()[1] > y_train.value_counts()[2] else 0
        print("Ternary mode selected. NEU and NONE will be both treated as NEU" if threshold is 1 else
              "Ternary mode selected. NEU and NONE will be both treated as NONE")
        y_train = [label - 1 if label > 1 else label for label in y_train]

    classifier.fit(x_train, y_train)
    predictions, probabilities = get_predictions(classifier, x_test)

    if reduced:
        predictions = [pred+1 if pred > threshold else pred for pred in predictions]

    score = None
    if y_test is not None:
        score = utils.print_confusion_matrix(predictions, y_test)
    return classifier, predictions, probabilities, score


def get_predictions(trained_classifier, feature_test_vector, is_vso=False):
    if is_vso:
        return trained_classifier.predict(feature_test_vector), trained_classifier.decision_function(feature_test_vector)
    else:
        return trained_classifier.predict(feature_test_vector), trained_classifier.predict_proba(feature_test_vector)


def get_model_accuracy(predictions, validation_labels):
    return metrics.accuracy_score(predictions, validation_labels)


if __name__ == '__main__':

    for S_DATASET in DATASET_ARRAY:

        print('################    DATASET: {}    ###############################'.format(S_DATASET))
        print()

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if B_UPSAMPLING:
            print('Performing upsampling...')
            train_data = perform_upsampling(train_data)

        # PRE-PROCESSING
        print('Data preprocessing...')
        train_data['preprocessed'] = preprocess(
            train_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
            bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
            bUsername=PREP_USERNAME, bURL=PREP_URL)
        dev_data['preprocessed'] = preprocess(
            dev_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
            bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
            bUsername=PREP_USERNAME, bURL=PREP_URL)
        if B_TEST_PHASE is True:
            test_data['preprocessed'] = preprocess(
                test_data['content'], bAll=PREP_ALL, bEmoji=PREP_EMOJI, bHashtags=PREP_HASHTAGS, bLaughter=PREP_LAUGHTER,
                bLetRep=PREP_LETREP, bLowercasing=PREP_LOWER, bNumber=PREP_NUMBER, bPunctuation=PREP_PUNCT, bXque=PREP_XQUE,
                bUsername=PREP_USERNAME, bURL=PREP_URL)

        # TOKENIZE
        print("Tokenizing...")
        train_data['tokenized'] = train_data.swifter.progress_bar(False).apply(
            lambda row: tokenize_sentence(row.preprocessed), axis=1)
        dev_data['tokenized'] = dev_data.swifter.progress_bar(False).apply(
            lambda row: tokenize_sentence(row.preprocessed), axis=1)
        if B_TEST_PHASE is True:
            test_data['tokenized'] = test_data.swifter.progress_bar(False).apply(
                lambda row: tokenize_sentence(row.preprocessed), axis=1)

        train_data['final_data'] = train_data['tokenized']
        dev_data['final_data'] = dev_data['tokenized']
        if B_TEST_PHASE is True:
            test_data['final_data'] = test_data['tokenized']

        # LIBREOFFICE CORRECTION
        if B_LIBREOFFICE:
            print("LibreOffice Processing... ")
            train_data['final_data'] = train_data.swifter.apply(lambda row: libreoffice_processing(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.apply(lambda row: libreoffice_processing(row.final_data), axis=1)
            if B_TEST_PHASE is True:
                test_data['final_data'] = test_data.swifter.apply(lambda row: libreoffice_processing(row.final_data), axis=1)

        # LEMMATIZING
        if B_LEMMATIZE:
            print("Lemmatizing data...")
            train_data['final_data'] = train_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)
            if B_TEST_PHASE is True:
                test_data['final_data'] = test_data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)

        # ACCENTS REMOVAL
        if B_REMOVE_ACCENTS:
            print("Removing accents...")
            train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
            if B_TEST_PHASE is True:
                test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)

        # STOPWORDS REMOVAL
        if B_REMOVE_STOPWORDS:
            print('Removing stopwords...')
            train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)
            dev_data['final_data'] = dev_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)
            if B_TEST_PHASE is True:
                test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)

        # FEATURE EXTRACTION
        print('Feature extraction...')
        train_features, dev_features, test_features = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if B_FEAT_LENGTH:
            train_features['tweet_length'] = extract_length_feature(train_data['content'])
            dev_features['tweet_length'] = extract_length_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['tweet_length'] = extract_length_feature(test_data['content'])

        if B_FEAT_UPPERCASE:
            train_features['has_uppercase'] = extract_uppercase_feature(train_data['content'])
            dev_features['has_uppercase'] = extract_uppercase_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['has_uppercase'] = extract_uppercase_feature(test_data['content'])

        if B_FEAT_QUESTIONMARK:
            train_features['question_mark'] = extract_question_mark_feature(train_data['content'])
            dev_features['question_mark'] = extract_question_mark_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['question_mark'] = extract_question_mark_feature(test_data['content'])

        if B_FEAT_EXCLAMARK:
            train_features['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
            dev_features['exclamation_mark'] = extract_exclamation_mark_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['exclamation_mark'] = extract_exclamation_mark_feature(test_data['content'])

        if B_FEAT_LET_REP:
            train_features['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
            dev_features['letter_repetition'] = extract_letter_repetition_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['letter_repetition'] = extract_letter_repetition_feature(test_data['content'])

        if B_FEAT_HASHTAGS:
            train_features['hashtag_number'] = extract_hashtag_number_feature(train_data['content'])
            dev_features['hashtag_number'] = extract_hashtag_number_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['hashtag_number'] = extract_hashtag_number_feature(test_data['content'])

        if B_FEAT_LAUGHTER:
            train_features['laughter_feature'] = extract_laughter_feature(train_data['content'])
            dev_features['laughter_feature'] = extract_laughter_feature(dev_data['content'])
            if B_TEST_PHASE:
                test_features['laughter_feature'] = extract_laughter_feature(test_data['content'])

        if B_FEAT_VOCABULARY:
            train_features['pos_voc'], train_features['neg_voc'], train_features['neu_voc'], train_features['none_voc'] = \
                extract_sent_words_feature(train_data['tokenized'], train_data['tokenized'], train_data['sentiment'],
                                           lexicons=B_LEXICONS, discriminating_terms=B_DISCRIMINATING_TERMS,
                                           discriminating_words=NUM_DISCRIMINATING_WORDS)
            dev_features['pos_voc'], dev_features['neg_voc'], dev_features['neu_voc'], dev_features['none_voc'] = \
                extract_sent_words_feature(dev_data['tokenized'], train_data['tokenized'], train_data['sentiment'],
                                           lexicons=B_LEXICONS, discriminating_terms=B_DISCRIMINATING_TERMS,
                                           discriminating_words=NUM_DISCRIMINATING_WORDS)
            if B_TEST_PHASE:
                test_features['pos_voc'], test_features['neg_voc'], test_features['neu_voc'], test_features['none_voc'] = \
                    extract_sent_words_feature(test_data['tokenized'], train_data['tokenized'], train_data['sentiment'],
                                               lexicons=B_LEXICONS, discriminating_terms=B_DISCRIMINATING_TERMS,
                                               discriminating_words=NUM_DISCRIMINATING_WORDS)

        # COUNT VECTORS
        if B_COUNTVECTORS:
            train_count_vectors, dev_count_vectors = perform_count_vectors(
                train_data['final_data'], dev_data['final_data'], print_oovs=True, classic_bow=B_CLASSIC_BOW)
            if B_TEST_PHASE:
                _, test_count_vectors = perform_count_vectors(
                    train_data['final_data'], test_data['final_data'], print_oovs=True, classic_bow=B_CLASSIC_BOW)

        # TF-IDF VECTORS
        if B_TFIDF:
            train_tfidf, dev_tfidf = perform_tf_idf_vectors(train_data['final_data'], dev_data['final_data'])
            if B_TEST_PHASE:
                _, test_tfidf = perform_tf_idf_vectors(train_data['final_data'], test_data['final_data'])

        # TRAINING
        set_name = 'FEATURES' if B_FEATURES else 'BOW'
        training_set = train_features if B_FEATURES else (train_count_vectors if B_COUNTVECTORS else train_tfidf)
        training_labels = train_data['sentiment']

        dev_set = dev_features if B_FEATURES else (dev_count_vectors if B_COUNTVECTORS else dev_tfidf)
        dev_labels = dev_data['sentiment']

        if B_TEST_PHASE is True:
            test_set = test_features if B_FEATURES else (test_count_vectors if B_COUNTVECTORS else test_tfidf)
            test_labels = test_data['sentiment']

        all_probabilities = []
        all_valid_probabilities = []
        all_test_probabilities = []

        print("MODEL:  {} ".format(set_name))
        mini, mini_test, mini_valid = [], [], []
        for j, clf in enumerate(CLASSIFIERS_ARRAY):

            if B_ONE_VS_REST:
                clf = OneVsRestClassifier(clf)
            print("Classifier: " + CLASSIFIERS_NAMES[j])
            print()

            print("Development set:")
            classif, preds, probs, train_score = train_model(
                clf, training_set, training_labels, dev_set, dev_labels, reduced=B_REDUCED)

            if B_TEST_PHASE:
                print("Test set:")
                test_preds, test_probs = get_predictions(classif, test_set)
                mini_test.append(test_probs)

            mini.append(probs)

        all_probabilities.append(mini)
        all_test_probabilities.append(mini_test)
        all_valid_probabilities.append(mini_valid)

        print("FASTTEXT MODEL")
        fasttext_path = '{}/{}_{}'.format(FT_MODEL_PATH, S_DATASET, FT_MODEL_NAME)
        fasttext_model = fasttext.load_model(path=fasttext_path)
        dev_fasttext_probabilities, dev_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
            fasttext_model, dev_data.content, label_dictionary)
        utils.print_confusion_matrix(dev_fasttext_predictions, dev_labels)
        test_fasttext_probabilities, test_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
            fasttext_model, test_data.content, label_dictionary)
        utils.print_confusion_matrix(test_fasttext_predictions, test_labels)

        # BERT
        print("BERT MODEL")
        bert_path = '{}/{}'.format(BERT_MODEL_PATH, BERT_MODEL_NAME)
        bert_model = TextClassifier.load(bert_path)
        dev_bert_probabilities, dev_bert_predictions = bert_embeddings.predict_with_bert_model(
            bert_model, dev_data.content, label_dictionary)
        utils.print_confusion_matrix(dev_bert_predictions, dev_labels)
        test_bert_probabilities, test_bert_predictions = bert_embeddings.predict_with_bert_model(
            bert_model, test_data.content, label_dictionary)
        utils.print_confusion_matrix(test_bert_predictions, test_labels)

        print('-------------- FINAL ENSEMBLE --------------')
        print()

        print('From Normal Models:')
        print()

        selected_classifier = 0  # See all the classifiers available above.
        probabilities_for_voting_ensemble_dev = [
            all_probabilities[0][selected_classifier],
            dev_fasttext_probabilities,
            dev_bert_probabilities
        ]

        if B_TEST_PHASE:
            probabilities_for_voting_ensemble_test = [
                all_test_probabilities[0][selected_classifier],
                test_fasttext_probabilities,
                test_bert_probabilities
            ]

        print(test_labels.size)
        utils.print_confusion_matrix(
            utils.get_averaged_predictions(probabilities_for_voting_ensemble_dev, len(dev_labels), label_dictionary),
            dev_labels)
        if B_TEST_PHASE:
            utils.print_confusion_matrix(
                utils.get_averaged_predictions(probabilities_for_voting_ensemble_test, len(test_labels), label_dictionary),
                test_labels)
        print()

        print()
        print('--------------- NEXT DATASET ------------------')

    print('-------------------------- THE END -----------------------------')
