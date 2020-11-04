import fasttext
from flair.models import TextClassifier
from src import utils, tweet_preprocessing, data_fetching, fasttext_embedding, bert_embeddings, feature_extraction
from src.config import *

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
warnings.simplefilter(action='ignore', category=Warning)


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


if __name__ == '__main__':

    for S_DATASET in DATASET_ARRAY:

        utils.print_separator('DATASET: {}'.format(S_DATASET))

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

        if B_UPSAMPLING:
            print('Performing upsampling...')
            train_data = tweet_preprocessing.perform_upsampling(train_data)

        # PRE-PROCESSING
        print('Data preprocessing...')
        train_data['preprocessed'] = tweet_preprocessing.preprocess_data(train_data['content'], 'main')
        dev_data['preprocessed'] = tweet_preprocessing.preprocess_data(dev_data['content'], 'main')
        if B_TEST_PHASE is True:
            test_data['preprocessed'] = tweet_preprocessing.preprocess_data(test_data['content'], 'main')

        # FEATURE EXTRACTION
        print('Feature extraction...')
        train_features = feature_extraction.extract_features(train_data, train_data)
        dev_features = feature_extraction.extract_features(dev_data, train_data)
        test_features = feature_extraction.extract_features(test_data, train_data)

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

        utils.print_separator("CLASSIC MODEL:  {} ".format(set_name))
        clf = CLASSIFIER
        print("Classifier: " + S_CLASSIFIER_NAME)
        clf.fit(training_set, training_labels)
        print("Development set:")
        dev_feat_predictions = clf.predict(dev_set)
        dev_feat_probabilities = clf.predict_proba(dev_set)
        utils.print_f1_score(dev_feat_predictions, dev_labels)
        if B_TEST_PHASE:
            print("Test set:")
            test_feat_predictions = clf.predict(test_set)
            test_feat_probabilities = clf.predict_proba(test_set)
            utils.print_f1_score(test_feat_predictions, test_labels)

        model_set = S_DATASET if not B_TRAIN_WITH_ALL else 'intertass'
        # FASTTEXT
        utils.print_separator("FASTTEXT MODEL")

        fasttext_path = '{}/{}_{}'.format(FT_MODEL_PATH, model_set, FT_MODEL_NAME)
        fasttext_model = fasttext.load_model(path=fasttext_path)
        dev_fasttext_probabilities, dev_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
            fasttext_model, dev_data.content, label_dictionary)
        print("Development set:")
        utils.print_f1_score(dev_fasttext_predictions, dev_labels)
        test_fasttext_probabilities, test_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
            fasttext_model, test_data.content, label_dictionary)
        print("Test set:")
        utils.print_f1_score(test_fasttext_predictions, test_labels)

        # BERT
        utils.print_separator("BERT MODEL")
        bert_path = '{}/{}/{}/best-model.pt'.format(BERT_MODEL_PATH, BERT_MODEL_NAME, model_set)
        bert_model = TextClassifier.load(bert_path)
        dev_bert_probabilities, dev_bert_predictions = bert_embeddings.predict_with_bert_model(
            bert_model, dev_data.content, label_dictionary)
        print("Development set:")
        utils.print_f1_score(dev_bert_predictions, dev_labels)
        test_bert_probabilities, test_bert_predictions = bert_embeddings.predict_with_bert_model(
            bert_model, test_data.content, label_dictionary)
        print("Test set:")
        utils.print_f1_score(test_bert_predictions, test_labels)

        utils.print_separator('FINAL ENSEMBLE')

        selected_classifier = 0  # See all the classifiers available above.
        probabilities_for_voting_ensemble_dev = [
            dev_feat_probabilities,
            dev_fasttext_probabilities,
            dev_bert_probabilities
        ]

        if B_TEST_PHASE:
            probabilities_for_voting_ensemble_test = [
                test_feat_probabilities,
                test_fasttext_probabilities,
                test_bert_probabilities
            ]

        print("Development set:")
        utils.print_f1_score(
            utils.get_averaged_predictions(probabilities_for_voting_ensemble_dev, len(dev_labels), label_dictionary),
            dev_labels)
        if B_TEST_PHASE:
            print("Test set:")
            utils.print_f1_score(
                utils.get_averaged_predictions(probabilities_for_voting_ensemble_test, len(test_labels), label_dictionary),
                test_labels)

        print()
        print('--------------- NEXT DATASET ------------------')

    print('-------------------------- THE END -----------------------------')
