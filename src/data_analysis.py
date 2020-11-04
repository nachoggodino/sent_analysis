from src.config import *
from src import utils, tweet_preprocessing, data_fetching, feature_extraction



# def isol_analytics(dataframe, feed, sentiment_feed):
#     pos_df = pandas.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
#     neg_df = pandas.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
#     pos_voc = pos_df['words'].array
#     neg_voc = neg_df['words'].array
#
#     pos_count, neg_count = set(), set()
#     for tweet in dataframe:
#         pos_count = pos_count.union(set(tweet) & set(pos_voc))
#         neg_count = neg_count.union(set(tweet) & set(neg_voc))
#     print('Ocurrencias en iSol:     NEG-> {}        POS-> {}'.format(len(neg_count), len(pos_count)))
#     print()
#
#     positive_voc, negative_voc = get_sentiment_vocabulary(feed, sentiment_feed, 3, 0)
#     pos_count.clear()
#     neg_count.clear()
#     for tweet in dataframe:
#         pos_count = pos_count.union(set(tweet) & set(positive_voc))
#         neg_count = neg_count.union(set(tweet) & set(negative_voc))
#     print('Ocurrencias en keyterms:     NEG-> {}        POS-> {}'.format(len(neg_count), len(pos_count)))
#     print()
#     print('Palabras discriminantes que no están en isol:')
#     print('POS-> {}'.format(numpy.setdiff1d(positive_voc, pos_voc)))
#     print('NEG-> {}'.format(numpy.setdiff1d(negative_voc, neg_voc)))
#
#
# def print_vocabulary_analysis(tokenized_train_list, tokenized_dev_list):
#     all_train_words = [item for sublist in tokenized_train_list for item in sublist]
#     all_dev_words = [item for sublist in tokenized_dev_list for item in sublist]
#     train_vocabulary = []
#     dev_vocabulary = []
#     for word in all_train_words:
#         if word not in train_vocabulary:
#             train_vocabulary.append(word)
#     for word in all_dev_words:
#         if word not in dev_vocabulary:
#             dev_vocabulary.append(word)
#     train_word_counter = Counter(all_train_words)
#     most_common_train_words = train_word_counter.most_common(10)
#     dev_word_counter = Counter(all_dev_words)
#     most_common_dev_words = dev_word_counter.most_common(10)
#     print("The total number of words in TRAINING_DATA is: " + str(len(all_train_words)))
#     print("The length of the vocabulary in TRAINING_DATA is: " + str(len(train_vocabulary)))
#     print("Most common words in TRAINING_DATA:")
#     print(most_common_train_words)
#     print()
#     print("The total number of words in DEVELOPMENT_DATA is: " + str(len(all_dev_words)))
#     print("The length of the vocabulary in DEVELOPMENT_DATA is: " + str(len(dev_vocabulary)))
#     print("Most common words in DEVELOPMENT_DATA:")
#     print(most_common_dev_words)
#     print()
#
#     out_of_vocabulary = []
#     for word in dev_vocabulary:
#         if word not in train_vocabulary:
#             out_of_vocabulary.append(word)
#     print("The number of Out-Of-Vocabulary words is: " + str(len(out_of_vocabulary)))
#     print("Which is the " + str(len(out_of_vocabulary) / len(dev_vocabulary) * 100) + "% of the Development Vocabulary")
#     print()
#
#
# def libreoffice_processing_analytics(dataframe, dictionary):
#     correction_count, nothing_count = set(), set()
#     for sentence in dataframe:
#         for word in sentence:
#             if dictionary.spell(word):
#                 nothing_count.add(word)
#             else:
#                 if word not in correction_count:
#                     print('{}           ---->           {}'.format(word, dictionary.suggest(word)))
#                     correction_count.add(word)
#                     print()
#     print('palabras corregidas:{}       palabras sin tocar:{}'.format(len(correction_count), len(nothing_count)))
#     return


if __name__ == '__main__':
    for S_DATASET in DATASET_ARRAY:

        utils.print_separator('DATASET: {}'.format(S_DATASET))

        print('Fetching the data...')
        train_data, dev_data, test_data, label_dictionary = data_fetching.fetch_data(S_DATASET)

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

        train_length = len(train_data)
        test_length = len(test_data)

        if B_TWEET_LENGTH_ANALYSIS:
            utils.print_separator('Tweet length analysis')
            print('Maximum lengths (words):')
            print('Average lengths (words):')
            print('     train: {}'.format(train_features['tweet_length'].max()))
            print('     test: {}'.format(test_features['tweet_length'].max()))

            print('Average lengths (words):')
            print('     train: {}'.format(train_features['tweet_length'].mean()))
            print('     test: {}'.format(test_features['tweet_length'].mean()))

        if B_HASHTAG_ANALYSIS:
            utils.print_separator('Hashtags analysis')
            print('Average number (hashtags/tweet):')
            print('     train: {}'.format(train_features['hashtag_number'].mean()))
            print('     test: {}'.format(test_features['hashtag_number'].mean()))

            print('Tweets containing hashtags:')
            print('     train: {}%'.format(train_features[train_features['hashtag_number'] > 0].count()['hashtag_number']
                                          * 100 / train_length))
            print('     test: {}%'.format(test_features[test_features['hashtag_number'] > 0].count()['hashtag_number']
                                         * 100 / test_length))




# VOCABULARY ANALYSIS

# print("The maximum length of a Tweet in TRAINING_DATA is: " + str(max(train_data['length'])))
# print("The minimum length of a Tweet in TRAINING_DATA is: " + str(min(train_data['length'])))
# print(
#     "The average length of a Tweet in TRAINING_DATA is: " + str(sum(train_data['length']) / len(train_data['length'])))
# print()
# print("The maximum length of a Tweet in DEVELOPMENT_DATA is: " + str(max(test_data['length'])))
# print("The minimum length of a Tweet in DEVELOPMENT_DATA is: " + str(min(test_data['length'])))
# print("The average length of a Tweet in DEVELOPMENT_DATA is: " + str(sum(test_data['length']) / len(test_data['length'])))
# print()

# print_separator("Vocabulary Analysis after tokenize")
#
# print_vocabulary_analysis(train_data['tokenized'], test_data['tokenized'])
#
# print_separator("Vocabulary Analysis after Libreoffice Processing")
# libreoffice_processing_analytics(train_data['tokenized'], dictionary)


# print("LibreOffice Processing... ")
# train_dev_data['final_data'] = train_dev_data.swifter.progress_bar(True).apply(
#     lambda row: utils.libreoffice_processing(row.tokenized, dictionary), axis=1)
# test_data['final_data'] = test_data.swifter.apply(lambda row: utils.libreoffice_processing(row.tokenized, dictionary),
#                                                   axis=1)

# print_separator("ISOL and discriminating analysis")
# isol_analytics(train_dev_data['tokenized'], train_dev_data['tokenized'], train_dev_data['sentiment'])
# isol_analytics(test_data['tokenized'], train_dev_data['tokenized'], train_dev_data['sentiment'])

# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("After lemmatizing the data...")
# print("Lemmatizing data...")
# train_data['final_data'] = train_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.final_data, lemmatizer), axis=1)
# test_data['final_data'] = test_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.final_data, lemmatizer), axis=1)
#
# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("After removing accents to the data...")
# print("Removing accents...")
# train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
# test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
#
# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("Label Counts:")
#
# print("Total count of each Label in TRAINING_DATA:")
# print(train_data['sentiment'].value_counts('N'))
# print()
# print("Total count of each Label in DEVELOPMENT_DATA:")
# print(dev_data['sentiment'].value_counts('N'))
# print()

# print_separator("Most discriminating words analysis")

# print("Training data:")
# # train_pos_voc, train_neg_voc = get_sentiment_vocabulary(train_data['tokenized'], 'P', 'N')
# train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data[
#     'none_voc'] = extract_sent_words_feature(train_data['tokenized'], train_data['tokenized'])
# print("The most discriminating words between P and N are:")
# print("Category P:")
# # print(train_pos_voc)
# print("Category N:")
# # print(train_neg_voc)
# print()
#
# print("Development data:")
# # dev_pos_voc, dev_neg_voc = get_sentiment_vocabulary(test_data['tokenized'], 'P', 'N')
# test_data['pos_voc'], test_data['neg_voc'], test_data['neu_voc'], test_data['none_voc'] = extract_sent_words_feature(
#     test_data['tokenized'], train_data['tokenized'])
# print("The most discriminating words between P and N are:")
# print("Category P:")
# # print(dev_pos_voc)
# print("Category N:")
# # print(dev_neg_voc)
# print()
#
# print_separator("Correlation analysis in TRAINING_DATA:")
#
# with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    #     '''
    #     print("Hour VS Sentiment")
    #     print()
    #     print(train_data.groupby(['hour', 'sentiment']).size())
    #     print("Month VS Sentiment")
    #     print()
    #     print(train_data.groupby(['month', 'sentiment']).size())
    #     print("Day of the Week VS Sentiment")
    #     print()
    #     print(train_data.groupby(['day_of_week', 'sentiment']).size())
    #     print("Length VS Sentiment")
    #     print()
    #     print(train_data.groupby(['length', 'sentiment']).size())
    #     '''

    # print("Laughter VS Sentiment")
    # print(train_data.groupby(['laughter_feature', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['length'], numpy.arange(0, 1+50, 5)), 'sentiment']).size())
    # print("Uppercase VS Sentiment")
    # print()
    # print(train_data.groupby(['has_uppercase', 'sentiment']).size())
    # print("Question VS Sentiment")
    # print()
    # print(train_data.groupby(['question_mark', 'sentiment']).size())
    # print("Exclamation VS Sentiment")
    # print()
    # print(train_data.groupby(['exclamation_mark', 'sentiment']).size())
    # print("Letter Repetition VS Sentiment")
    # print()
    # print(train_data.groupby(['letter_repetition', 'sentiment']).size())
    # print("Positive Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['pos_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Negative Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['neg_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Neutral Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['neu_voc'], numpy.arange(0, 1.0+4, 0.2)), 'sentiment']).size())
    # print("None Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['none_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    #
    print("Correlation analysis in DEVELOPMENT_DATA:")
    # print()

    print("Laughter VS Sentiment")
    print()
    print(test_data.groupby(['laughter_feature', 'sentiment']).size())

    # '''
    # print("Hour VS Sentiment")
    # print()
    # print(dev_data.groupby(['hour', 'sentiment']).size())
    # print("Month VS Sentiment")
    # print()
    # print(dev_data.groupby(['month', 'sentiment']).size())
    # print("Day of the Week VS Sentiment")
    # print()
    # print(dev_data.groupby(['day_of_week', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(dev_data.groupby(['length', 'sentiment']).size())
    # '''
    # print("Uppercase VS Sentiment")
    # print()
    # print(test_data.groupby(['has_uppercase', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['length'], numpy.arange(0, 1+50, 5)), 'sentiment']).size())
    # print("Question VS Sentiment")
    # print()
    # print(test_data.groupby(['question_mark', 'sentiment']).size())
    # print("Exclamation VS Sentiment")
    # print()
    # print(test_data.groupby(['exclamation_mark', 'sentiment']).size())
    # print("Positive Vocabulary VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['pos_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Negative Vocabulary VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['neg_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Neutral Vocabulary VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['neu_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("None Vocabulary VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['none_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
